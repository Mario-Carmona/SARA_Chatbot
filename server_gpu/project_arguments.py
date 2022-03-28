
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProyectArguments:
    workdir: str = field(
        metadata={
            "help": ""
        }
    )
