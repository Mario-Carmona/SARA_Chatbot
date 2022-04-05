
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProyectArguments:
    workdir: str = field(
        metadata={
            "help": ""
        }
    )
    host: str = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    port: str = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    controller_url: str = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    ngrok_path: str = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    ngrok_config_path: str = field(
        default=None,
        metadata={
            "help": ""
        }
    )
