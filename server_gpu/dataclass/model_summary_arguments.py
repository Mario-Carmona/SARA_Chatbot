
import os
from dataclasses import dataclass, field


@dataclass
class ModelSumArguments:
    """
    Argumentos relacionados con el modelo usado para resumir textos
    """

    model_summary: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_summary_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_summary_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_summary_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    def __post_init__(self):
        assert os.path.exists(self.model_summary), "`model_summary` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_config), "`model_summary_config` debe ser un archivo existente."
        assert os.path.exists(self.model_summary_tokenizer), "`model_summary_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_tokenizer_config), "`model_summary_tokenizer_config` debe ser un archivo existente."
