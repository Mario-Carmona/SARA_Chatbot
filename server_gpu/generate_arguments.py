
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerateArguments:
    do_sample: Optional[bool] = field(
        metadata={
            "help": ""
        }
    )
    temperature: Optional[float] = field(
        metadata={
            "help": ""
        }
    )
    top_p: Optional[float] = field(
        metadata={
            "help": ""
        }
    )
    max_time: Optional[float] = field(
        metadata={
            "help": ""
        }
    )
    max_length: Optional[int] = field(
        metadata={
            "help": ""
        }
    )
    min_length: Optional[int] = field(
        metadata={
            "help": ""
        }
    )
    use_cache: Optional[bool] = field(
        metadata={
            "help": ""
        }
    )
