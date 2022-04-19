#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cProfile import label
from pathlib import Path
import argparse
import json
import sys
import os
import logging
import linecache
from typing import Dict, Tuple, List, Callable, Iterable
from color import bcolors

from torch.utils.data import Dataset, Sampler
from datasets import load_dataset, load_metric
from torch.nn import functional as F

from dataclass.finetuning_arguments import FinetuningArguments
from transformers import HfArgumentParser
from transformers import Seq2SeqTrainingArguments

from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer, BartTokenizer

import transformers
from transformers import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, BlenderbotForConditionalGeneration
from transformers import EvalPrediction, Trainer, Seq2SeqTrainer
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from transformers.training_args import ParallelMode

import torch
from torch import nn

from sacrebleu import corpus_bleu

import numpy as np

from torch.utils.data import DataLoader

import pickle

from transformers.file_utils import cached_property

import torch.distributed as dist

import math



from torch.utils.data import Dataset



try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False





def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)



def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data




def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))



class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")




class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding





def main():

    logger = logging.getLogger(__name__)

    def check_output_dir(args, expected_items=0):
        """
        Checks whether to bail out if output_dir already exists and has more than expected_items in it
        `args`: needs to have the following attributes of `args`:
        - output_dir
        - do_train
        - overwrite_output_dir
        `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
        """
        if (
            os.path.exists(args.output_dir)
            and len(os.listdir(args.output_dir)) > expected_items
            and args.do_train
            and not args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and "
                f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
                "Use --overwrite_output_dir to overcome."
            )


    def save_json(content, path, indent=4, **json_dump_kwargs):
        with open(path, "w") as f:
            json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


    def handle_metrics(split, metrics, output_dir):
        """
        Log and save metrics
        Args:
        - split: one of train, val, test
        - metrics: metrics dict
        - output_dir: where to save the metrics
        """

        logger.info(bcolors.OK + f"***** {split} metrics *****" + bcolors.RESET)
        for key in sorted(metrics.keys()):
            logger.info(bcolors.OK + f"  {key} = {metrics[key]}" + bcolors.RESET)
        save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


    def freeze_params(model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False


    def freeze_embeds(model):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        model_type = model.config.model_type

        if model_type == "t5":
            freeze_params(model.shared)
            for d in [model.encoder, model.decoder]:
                freeze_params(d.embed_tokens)
        elif model_type == "fsmt":
            for d in [model.model.encoder, model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        else:
            freeze_params(model.model.shared)
            for d in [model.model.encoder, model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

    def lmap(f: Callable, x: Iterable) -> List:
        """list(map(f, x))"""
        return list(map(f, x))

    def grad_status(model: nn.Module) -> Iterable:
        return (par.requires_grad for par in model.parameters())

    def assert_all_frozen(model):
        model_grads: List[bool] = list(grad_status(model))
        n_require_grad = sum(lmap(int, model_grads))
        npars = len(model_grads)
        assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


    def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
        """Uses sacrebleu's corpus_bleu implementation."""
        return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


    def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
        def non_pad_len(tokens: np.ndarray) -> int:
            return np.count_nonzero(tokens != tokenizer.pad_token_id)

        def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
            pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
            pred_str = lmap(str.strip, pred_str)
            label_str = lmap(str.strip, label_str)
            return pred_str, label_str

        def translation_metrics(pred: EvalPrediction) -> Dict:
            pred_str, label_str = decode_pred(pred)
            bleu: Dict = calculate_bleu(pred_str, label_str)
            gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
            bleu.update({"gen_len": gen_len})
            return bleu

        compute_metrics_fn = translation_metrics
        return compute_metrics_fn


















    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file", 
        type = str,
        help = "El formato del archivo debe ser \'config.json\'"
    )

    try:
        args = parser.parse_args()
        assert args.config_file.split('.')[-1] == "json"
    except:
        parser.print_help()
        sys.exit(0)


    BASE_PATH = Path(__file__).resolve().parent
    CONFIG_FILE = args.config_file


    parser = HfArgumentParser(
        (
            FinetuningArguments,
            Seq2SeqTrainingArguments
        )
    )

    finetuning_args, training_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    WORKDIR = finetuning_args.workdir

    training_args.output_dir = os.path.join(WORKDIR, training_args.output_dir)


    check_output_dir(training_args)


    # Ruta donde instalar las extensiones de Pytorch
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(WORKDIR, "torch_extensions")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        bcolors.WARNING + "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" + bcolors.RESET,
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(bcolors.OK + "Training/evaluation parameters %s" + bcolors.RESET, training_args)





    # Set seed before initializing model.
    set_seed(training_args.seed)





    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    configConver = AutoConfig.from_pretrained(
        finetuning_args.model_conver_config,
        task_specific_params={
            finetuning_args.task: {
                "do_sample": finetuning_args.do_sample,
                "temperature": finetuning_args.temperature,
                "top_p": finetuning_args.top_p,
                "max_length": finetuning_args.max_length,
                "min_length": finetuning_args.min_length
            }
        }
    )

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(configConver, p), f"({configConver.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(configConver, p, getattr(training_args, p))

    tokenizerConver = AutoTokenizer.from_pretrained(
        finetuning_args.model_conver_tokenizer,
        config=finetuning_args.model_conver_tokenizer_config,
        use_fast=True
    )

    modelConver = AutoModelForSeq2SeqLM.from_pretrained(
        finetuning_args.model_conver,
        from_tf=bool(".ckpt" in finetuning_args.model_conver),
        config=configConver,
        torch_dtype=torch.float16
    )



    if True:
        freeze_embeds(modelConver)
    if True:
        freeze_params(modelConver.get_encoder())
        assert_all_frozen(modelConver.get_encoder())



    dataset_class = Seq2SeqDataset


    train_dataset = (
        dataset_class(
            tokenizerConver,
            type_path="train",
            data_dir=finetuning_args.data_dir,
            n_obs=finetuning_args.n_train,
            max_target_length=finetuning_args.max_target_length,
            max_source_length=finetuning_args.max_source_length,
            prefix=modelConver.config.prefix or "",
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizerConver,
            type_path="val",
            data_dir=finetuning_args.data_dir,
            n_obs=finetuning_args.n_val,
            max_target_length=128,
            max_source_length=finetuning_args.max_source_length,
            prefix=modelConver.config.prefix or "",
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )


    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizerConver.pad_token_id)


    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizerConver.batch_decode(pred.predictions[0], skip_special_tokens=True)
        label_str = tokenizerConver.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str


    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu


    metric = load_metric("accuracy")

    def compute_metrics(eval_pred: EvalPrediction):
        # No se si es el índice 0 ó 1, se podrá comprobar cuando
        # se tengan más datos porque no se si es la predicción
        # ó la máscara. Parece que es el cero porque la tercera
        # dimensión es igual a 8008 al igual que logits en la versión
        # de Pytorch y es igual al tamaño del vocabulario del modelo
        predictions = np.argmax(eval_pred.predictions[0], axis=-1)
        predictions = predictions.flatten()
        references = eval_pred.label_ids.flatten()
        return metric.compute(predictions=predictions, references=references)



    trainer = Seq2SeqTrainer(
        model=modelConver,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizerConver,
        data_collator=Seq2SeqDataCollator(tokenizerConver, finetuning_args, training_args.tpu_num_cores),
        compute_metrics=compute_metrics
    )


    os.system("nvidia-smi")


    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info(bcolors.OK + "*** Train ***" + bcolors.RESET)

        train_result = trainer.train(
            model_path=finetuning_args.model_conver
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = finetuning_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizerConver.save_pretrained(training_args.output_dir)


    # Evaluation
    if training_args.do_eval:
        logger.info(bcolors.OK + "*** Evaluate ***" + bcolors.RESET)

        metrics = trainer.evaluate(
            metric_key_prefix="val"
        )
        metrics["val_n_objs"] = finetuning_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():

            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)


    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics



if __name__ == "__main__":
    main()
