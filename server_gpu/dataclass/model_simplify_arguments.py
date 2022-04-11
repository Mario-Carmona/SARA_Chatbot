
import os
from dataclasses import dataclass, field


@dataclass
class ModelSimplifyArguments:
    """
    Argumentos relacionados con el modelo usado para simplificar textos
    """
    
    model_simplify: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_simplify_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_simplify_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_simplify_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    def __post_init__(self):
        assert os.path.exists(self.model_simplify), "`model_simplify` debe ser un directorio existente."
        assert os.path.exists(self.model_simplify_config), "`model_simplify_config` debe ser un archivo existente."
        assert os.path.exists(self.model_simplify_tokenizer), "`model_simplify_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_simplify_tokenizer_config), "`model_simplify_tokenizer_config` debe ser un archivo existente."
