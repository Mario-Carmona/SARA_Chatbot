
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

    def __post_init__(self):
        assert os.path.exists(self.workdir), "`workdir` debe ser un directorio existente."
