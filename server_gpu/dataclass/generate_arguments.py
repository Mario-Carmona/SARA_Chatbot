
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerateArguments:
    max_length: int = field(
        metadata={
            "help": ""
        }
    )
    min_length: int = field(
        metadata={
            "help": ""
        }
    )
