
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProyectArguments:
    workdir: str = field(
        metadata={
            "help": ""
        }
    )
    host: Optional[str] = field(
        metadata={
            "help": ""
        }
    )
    port: Optional[str] = field(
        metadata={
            "help": ""
        }
    )
    controller_url: Optional[str] = field(
        metadata={
            "help": ""
        }
    )
    ngrok_path: Optional[str] = field(
        metadata={
            "help": ""
        }
    )
    ngrok_config_path: Optional[str] = field(
        metadata={
            "help": ""
        }
    )
