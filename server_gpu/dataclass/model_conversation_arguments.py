
import os
from dataclasses import dataclass, field
from project_arguments import ProyectArguments


@dataclass
class ModelConverArguments(ProyectArguments):
    """
    Argumentos relacionados con el modelos usado para realizar la conversación
    """
    
    model_conver: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_conver_config: str = field(
        default=None,
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_conver_tokenizer: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_conver_tokenizer_config: str = field(
        default=None,
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    def __post_init__(self):
        self.model_conver = self.workdir + self.model_conver
        self.model_conver_config = self.workdir + self.model_conver_config
        self.model_conver_tokenizer = self.workdir + self.model_conver_tokenizer
        self.model_conver_tokenizer_config = self.workdir + self.model_conver_tokenizer_config

        assert os.path.exists(self.model_conver), "`model_conver` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_config), "`model_conver_config` debe ser un archivo existente."
        assert os.path.exists(self.model_conver_tokenizer), "`model_conver_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_tokenizer_config), "`model_conver_tokenizer_config` debe ser un archivo existente."
