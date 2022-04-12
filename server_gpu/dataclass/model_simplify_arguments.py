
import os
from dataclasses import dataclass, field
from SentenceSimplification.muss.simplify import ALLOWED_MODEL_NAMES


@dataclass
class ModelSimplifyArguments:
    """
    Argumentos relacionados con el modelo usado para simplificar textos
    """
    
    model_simplify: str = field(
        metadata={
            "help": ""
        }
    )

    def __post_init__(self):
        assert self.model_simplify in ALLOWED_MODEL_NAMES, "`model_simplify` debe ser uno de los modelo preentrenados disponibles."
