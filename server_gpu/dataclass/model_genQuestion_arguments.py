
import os
from dataclasses import dataclass, field


@dataclass
class ModelGenQuestionArguments:
    """
    Argumentos relacionados con el modelo usado para generar preguntas
    """
    
    model_genQuestion: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_genQuestion_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_genQuestion_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_genQuestion_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    def __post_init__(self):
        assert os.path.exists(self.model_genQuestion), "`model_genQuestion` debe ser un directorio existente."
        assert os.path.exists(self.model_genQuestion_config), "`model_genQuestion_config` debe ser un archivo existente."
        assert os.path.exists(self.model_genQuestion_tokenizer), "`model_genQuestion_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_genQuestion_tokenizer_config), "`model_genQuestion_tokenizer_config` debe ser un archivo existente."
