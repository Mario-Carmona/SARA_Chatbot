
import os
from typing import List
from dataclasses import dataclass, field
from dataclass.project_arguments import ProyectArguments


@dataclass
class ExtractSentimentsArguments(ProyectArguments):
    """
    Argumentos relacionados con la extracción de ejemplos del dataset `empathetic_dialogues`
    """

    ALLOWED_SENTIMENTS = [
        'sad'
    ]

    list_sentiment: List[str] = field(
        metadata={
            "help": ""
        }
    )
    num_samples: int = field(
        metadata={
            "help": ""
        }
    )
    train_split: float = field(
        metadata={
            "help": ""
        }
    )
    result_dir: str = field(
        metadata={
            "help": ""
        }
    )

    def __post_init__(self):
        self.result_file = self.workdir + self.result_file

        assert len(self.list_sentiment) > 0, "`list_sentiment` debe ser una lista con al menos un elemento"
        for sentiment in self.list_sentiment:
            assert sentiment in self.ALLOWED_SENTIMENTS, f"`list_sentiment` debe contener elementos válidos: {self.ALLOWED_SENTIMENTS}"

        assert self.num_samples >= 0, "`num_samples` debe ser un entero positivo."

        assert 0.0 < self.train_split and self.train_split < 1.0, "`train_split` debe estar en el rango (0,1)."

        assert os.path.exists(self.result_dir), "`result_dir` debe ser un directorio existente."
