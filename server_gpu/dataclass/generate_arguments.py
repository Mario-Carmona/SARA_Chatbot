
from dataclasses import dataclass, field
from typing import Optional

from pkg_resources import require


@dataclass
class GenerateArguments:
    do_sample: bool = field(
        metadata={
            "help": ""
        },
        required=False
    )
    temperature: float = field(
        metadata={
            "help": ""
        },
        required=False
    )
    top_p: float = field(
        metadata={
            "help": ""
        },
        required=False
    )
    max_time: float = field(
        metadata={
            "help": ""
        },
        required=False
    )
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
    use_cache: bool = field(
        metadata={
            "help": ""
        },
        required=False
    )
