
from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class Seq2SeqTrainingArgumentsCustom(Seq2SeqTrainingArguments):
    do_search_hyperparam: bool = field(
        default=False,
        metadata={
            "help": ""
        }
    )














