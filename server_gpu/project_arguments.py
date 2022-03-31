
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
        metadata={
            "help": ""
        }
    )
    port: str = field(
        metadata={
            "help": ""
        }
    )
    controller_url: str = field(
        metadata={
            "help": ""
        }
    )
    ngrok_path: str = field(
        metadata={
            "help": ""
        }
    )
    ngrok_config_path: str = field(
        metadata={
            "help": ""
        }
    )
