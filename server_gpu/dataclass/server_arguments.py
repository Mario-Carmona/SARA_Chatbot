
import os
from dataclasses import dataclass, field
from dataclass.model_conversation_arguments import ModelConverArguments
from dataclass.deepl_arguments import DeeplArguments
from dataclass.generate_arguments import GenerateArguments


@dataclass
class ServerArguments(ModelConverArguments, DeeplArguments, 
                      GenerateArguments):
    """
    Argumentos relacionados con la configuración del servidor
    """

    host: str = field(
        metadata={
            "help": "Dirección host del servidor GPU"
        }
    )
    port: str = field(
        metadata={
            "help": "Puerto del servidor GPU"
        }
    )
    controller_url: str = field(
        metadata={
            "help": "Dirección URL del servidor del controlador"
        }
    )
    ngrok_path: str = field(
        metadata={
            "help": "Ruta al ejecutable de ngrok"
        }
    )
    ngrok_config_path: str = field(
        metadata={
            "help": "Ruta al archivo de configuración de ngrok"
        }
    )
    auth_key_deepl: str = field(
        metadata={
            "help": "Clave de autenticación para la API de DeepL"
        }
    )  

    def __post_init__(self):
        ModelConverArguments.__post_init__(self)

        self.ngrok_path = self.workdir + self.ngrok_path
        self.ngrok_config_path = self.workdir + self.ngrok_config_path

        assert os.path.exists(self.ngrok_path), "`ngrok_path` debe ser un archivo existente."

        assert os.path.exists(self.ngrok_config_path), "`ngrok_config_path` debe ser un directorio existente."
        assert self.ngrok_config_path.split(".")[-1] == "yml", "`ngrok_config_path` debe ser un archivo YAML."
