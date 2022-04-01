from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetArguments:
    dataset_name: str = field(
        metadata={
            "help": ""
        }
    )
    train_dataset_name: str = field(
        metadata={
            "help": ""
        }
    )
    test_dataset_name: str = field(
        metadata={
            "help": ""
        }
    )
