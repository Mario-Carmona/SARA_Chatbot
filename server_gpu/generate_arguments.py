
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerateArguments:
    do_sample: bool = field(
        metadata={
            "help": ""
        }
    )
    temperature: float = field(
        metadata={
            "help": ""
        }
    )
    top_p: float = field(
        metadata={
            "help": ""
        }
    )
    max_time: float = field(
        metadata={
            "help": ""
        }
    )
    max_length: int = field(
        metadata={
            "help": ""
        }
    )
    use_cache: bool = field(
        metadata={
            "help": ""
        }
    )
