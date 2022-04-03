
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    train_file: Optional[bool] = field(
        metadata={
            "help": ""
        }
    )
    validation_file: Optional[bool] = field(
        metadata={
            "help": ""
        }
    )
    max_seq_length: Optional[int] = field(
        metadata={
            "help": ""
        }
    )
    pad_to_max_length: Optional[bool] = field(
        metadata={
            "help": ""
        }
    )
    doc_stride: Optional[int] = field(
        metadata={
            "help": ""
        }
    )
