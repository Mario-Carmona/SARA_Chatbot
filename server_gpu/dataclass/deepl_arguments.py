
import os
from dataclasses import dataclass, field


@dataclass
class DeeplArguments:
    """
    Argumentos relacionados con el módulo traductor de DeepL
    """
    
    auth_key_deepl: str = field(
        default=None,
        metadata={
            "help": "Clave de autenticación para la API de DeepL"
        }
    )
