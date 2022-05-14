
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
