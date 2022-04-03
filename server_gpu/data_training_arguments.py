
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    train_file: bool = field(
        metadata={
            "help": ""
        }
    )
    validation_file: bool = field(
        metadata={
            "help": ""
        }
    )
    max_seq_length: int = field(
        metadata={
            "help": ""
        }
    )
    pad_to_max_length: bool = field(
        metadata={
            "help": ""
        }
    )
    doc_stride: int = field(
        metadata={
            "help": ""
        }
    )
