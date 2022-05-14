
import os
from dataclasses import dataclass, field
from dataclass.deepl_arguments import DeeplArguments
from dataclass.generate_arguments import GenerateArguments


@dataclass
class ServerArguments(DeeplArguments, GenerateArguments):
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
    model_conver_adult: str = field(
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_conver_adult_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_conver_adult_tokenizer: str = field(
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_conver_adult_tokenizer_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )
    model_conver_child: str = field(
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_conver_child_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_conver_child_tokenizer: str = field(
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_conver_child_tokenizer_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )
    tam_history: int = field(
        metadata={
            "help": "Número máximo de tokens que forman el historial de la conversación"
        }
    )

    def __post_init__(self):
        self.ngrok_path = os.path.join(self.workdir, self.ngrok_path)
        self.ngrok_config_path = os.path.join(self.workdir, self.ngrok_config_path)

        self.model_conver_adult = os.path.join(self.workdir, self.model_conver_adult)
        self.model_conver_adult_config = os.path.join(self.workdir, self.model_conver_adult_config)
        self.model_conver_adult_tokenizer = os.path.join(self.workdir, self.model_conver_adult_tokenizer)
        self.model_conver_adult_tokenizer_config = os.path.join(self.workdir, self.model_conver_adult_tokenizer_config)

        self.model_conver_child = os.path.join(self.workdir, self.model_conver_child)
        self.model_conver_child_config = os.path.join(self.workdir, self.model_conver_child_config)
        self.model_conver_child_tokenizer = os.path.join(self.workdir, self.model_conver_child_tokenizer)
        self.model_conver_child_tokenizer_config = os.path.join(self.workdir, self.model_conver_child_tokenizer_config)

        assert os.path.exists(self.ngrok_path), "`ngrok_path` debe ser un archivo existente."

        assert os.path.exists(self.ngrok_config_path), "`ngrok_config_path` debe ser un archivo existente."
        assert self.ngrok_config_path.split(".")[-1] == "yml", "`ngrok_config_path` debe ser un archivo YAML."

        assert os.path.exists(self.model_conver_adult), "`model_conver_adult` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_adult_config), "`model_conver_adult_config` debe ser un archivo existente."
        assert os.path.exists(self.model_conver_adult_tokenizer), "`model_conver_adult_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_adult_tokenizer_config), "`model_conver_adult_tokenizer_config` debe ser un archivo existente."

        assert os.path.exists(self.model_conver_child), "`model_conver_child` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_child_config), "`model_conver_child_config` debe ser un archivo existente."
        assert os.path.exists(self.model_conver_child_tokenizer), "`model_conver_child_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_child_tokenizer_config), "`model_conver_child_tokenizer_config` debe ser un archivo existente."

        assert self.tam_history > 0, "`tam_history` debe ser un entero mayor que 0."
