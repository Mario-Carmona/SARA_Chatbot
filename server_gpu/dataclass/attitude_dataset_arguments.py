
import os
from typing import List
from dataclasses import dataclass, field
from dataclass.project_arguments import ProyectArguments


@dataclass
class AttitudeDatasetArguments(ProyectArguments):
    """
    Argumentos relacionados con la extracción de ejemplos del dataset `empathetic_dialogues`
    """

    ALLOWED_SENTIMENTS = [
        'content', 
        'angry', 
        'grateful', 
        'ashamed', 
        'annoyed', 
        'sentimental', 
        'surprised', 
        'excited', 
        'caring', 
        'anticipating', 
        'afraid', 
        'prepared', 
        'devastated', 
        'faithful', 
        'lonely', 
        'disappointed', 
        'sad', 
        'nostalgic', 
        'jealous', 
        'joyful', 
        'hopeful', 
        'trusting', 
        'disgusted', 
        'anxious', 
        'impressed', 
        'furious', 
        'proud', 
        'guilty', 
        'embarrassed', 
        'terrified', 
        'apprehensive', 
        'confident'
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
    attitude_dataset_file: str = field(
        metadata={
            "help": ""
        }
    )
    seed: int = field(
        metadata={
            "help": ""
        }
    )

    def __post_init__(self):
        self.attitude_dataset_file = os.path.join(self.workdir, self.attitude_dataset_file)

        assert len(self.list_sentiment) > 0, "`list_sentiment` debe ser una lista con al menos un elemento"
        for sentiment in self.list_sentiment:
            assert sentiment in self.ALLOWED_SENTIMENTS, f"`list_sentiment` debe contener elementos válidos: {self.ALLOWED_SENTIMENTS}"

        assert self.num_samples >= 0, "`num_samples` debe ser un entero positivo."

        dir = '/'.join(self.attitude_dataset_file.split('/')[:-1])
        assert os.path.exists(dir), "`attitude_dataset_file` debe estar en un directorio existente."
        assert self.attitude_dataset_file.split('.')[-1] == 'csv', "`attitude_dataset_file` debe ser un archivo CSV"
