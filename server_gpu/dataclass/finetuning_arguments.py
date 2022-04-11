
import os
from dataclasses import dataclass, field
from dataclass.model_conversation_arguments import ModelConverArguments
from dataclass.generate_arguments import GenerateArguments
from transformers import Seq2SeqTrainingArguments


@dataclass
class FinetuningArguments(ModelConverArguments, GenerateArguments, 
                          Seq2SeqTrainingArguments):
    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    train_dataset: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    validation_dataset: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    task: str = field(
        metadata={
            "help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"
        }
    )
    max_source_length: int = field(
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    max_target_length: int = field(
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    val_max_target_length: int = field(
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        }
    )
    n_train: int = field(
        default=-1, 
        metadata={
            "help": "# training examples. -1 means use all."
        }
    )
    n_val: int = field(
        default=-1, 
        metadata={
            "help": "# validation examples. -1 means use all."
        }
    )
    src_lang: str = field(
        default=None, 
        metadata={
            "help": "Source language id for translation."
        }
    )
    tgt_lang: str = field(
        default=None, 
        metadata={
            "help": "Target language id for translation."
        }
    )
    eval_beams: int = field(
        default=None, 
        metadata={
            "help": "# num_beams to use for evaluation."
        }
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."
        }
    )

    def __post_init__(self):
        ModelConverArguments.__post_init__(self)
        Seq2SeqTrainingArguments.__post_init__(self)

        self.output_dir = self.workdir + self.output_dir

        self.data_dir = self.workdir + self.data_dir
        self.train_dataset = self.data_dir + self.train_dataset
        self.validation_dataset = self.data_dir + self.validation_dataset

        assert os.path.exists(self.data_dir), "`data_dir` debe ser un directorio existente."

        assert os.path.exists(self.train_dataset), "`train_dataset` debe ser un archivo existente."
        assert self.train_dataset.split(".")[-1] == "csv", "`train_dataset` debe ser un archivo CSV."

        assert os.path.exists(self.validation_dataset), "`validation_dataset` debe ser un archivo existente."
        assert self.validation_dataset.split(".")[-1] == "csv", "`validation_dataset` debe ser un archivo CSV."

        assert self.max_source_length > 0, "`max_source_length` debe ser un entero positivo."

        assert self.max_target_length > 0, "`max_target_length` debe ser un entero positivo."

        assert self.val_max_target_length > 0, "`val_max_target_length` debe ser un entero positivo."
