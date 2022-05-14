
from dataclasses import dataclass, field
from typing import Optional

from pkg_resources import require


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
    do_sample: bool = field(
        metadata={
            "help": ""
        },
        default=None
    )
    temperature: float = field(
        metadata={
            "help": ""
        },
        default=None
    )
    top_p: float = field(
        metadata={
            "help": ""
        },
        default=None
    )
    max_time: float = field(
        metadata={
            "help": ""
        },
        default=None
    )
    use_cache: bool = field(
        metadata={
            "help": ""
        },
        default=None
    )
