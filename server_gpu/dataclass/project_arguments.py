
import os
from dataclasses import dataclass, field


@dataclass
class ProyectArguments:
    """
    Argumentos relacionados con aspectos generales del proyecto
    """

    workdir: str = field(
        metadata={
            "help": "Directorio de trabajo del proyecto"
        }
    )
    host: str = field(
        default=None,
        metadata={
            "help": "Dirección host del servidor GPU"
        }
    )
    port: str = field(
        default=None,
        metadata={
            "help": "Puerto del servidor GPU"
        }
    )
    controller_url: str = field(
        default=None,
        metadata={
            "help": "Dirección URL del servidor del controlador"
        }
    )
    ngrok_path: str = field(
        default=None,
        metadata={
            "help": "Ruta al ejecutable de ngrok"
        }
    )
    ngrok_config_path: str = field(
        default=None,
        metadata={
            "help": "Ruta al archivo de configuración de ngrok"
        }
    )
    auth_key_deepl: str = field(
        default=None,
        metadata={
            "help": "Clave de autenticación para la API de DeepL"
        }
    )

    def __post_init__(self):
        assert os.path.exists(self.workdir), "`workdir` debe ser un directorio existente."

        assert os.path.exists(self.ngrok_path), "`ngrok_path` debe ser un archivo existente."

        assert os.path.exists(self.ngrok_config_path), "`ngrok_config_path` debe ser un directorio existente."
        assert self.ngrok_config_path.split(".")[-1] == "yml", "`ngrok_config_path` debe ser un archivo YAML."
