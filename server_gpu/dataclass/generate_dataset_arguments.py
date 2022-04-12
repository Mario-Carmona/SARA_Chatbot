
import os
from dataclasses import dataclass, field

from dataclass.model_summary_arguments import ModelSumArguments
from dataclass.model_simplify_arguments import ModelSimplifyArguments
from dataclass.deepl_arguments import DeeplArguments
from dataclass.model_genQuestion_arguments import ModelGenQuestionArguments


@dataclass
class GenerateDatasetArguments(ModelSumArguments, ModelGenQuestionArguments,
                               ModelSimplifyArguments, DeeplArguments):
    """
    Argumentos relacionados con la generación de datasets para el entrenamiento
    """
    
    seed: int = field(
        metadata={
            "help": "Semilla del generador aleatorio"
        }
    )
    limit_summary: int = field(
        metadata={
            "help": "Mínimo número de tokens necesario para realizar el resumen"
        }
    )
    max_length_summary: int = field(
        metadata={
            "help": "Máxima longitud en tokens de las preguntas"
        }
    )
    max_length_question: int = field(
        metadata={
            "help": "Máxima longitud en tokens de las preguntas"
        }
    )
    num_beams_summary: int = field(
        metadata={
            "help": ""
        }
    )
    num_beams_question: int = field(
        metadata={
            "help": ""
        }
    )
    dataset_file: str = field(
        metadata={
            "help": "Archivo CSV que contiene el dataset"
        }
    )
    translated: bool = field(
        metadata={
            "help": "Indica si `dataset_file` es un archivo traducido"
        }
    )
    result_dir: str = field(
        metadata={
            "help": "Directorio destino de los archivos generados"
        }
    )
    train_split: float = field(
        metadata={
            "help": "Porcentaje de training obtenido del dataset"
        }
    )

    def __post_init__(self):
        ModelSumArguments.__post_init__(self)
        ModelGenQuestionArguments.__post_init__(self)
        ModelSimplifyArguments.__post_init__(self)

        self.dataset_file = self.workdir + self.dataset_file
        self.result_dir = self.workdir + self.result_dir

        assert self.limit_summary >= 0, "`limit_summary` debe ser un entero positivo."

        assert self.max_length_question >= 0, "`max_length_question` debe ser un entero positivo."

        assert self.num_beams_summary >= 0, "`num_beams_summary` debe ser un entero positivo."

        assert self.num_beams_question >= 0, "`num_beams_question` debe ser un entero positivo."

        assert os.path.exists(self.dataset_file), "`dataset_file` debe ser un archivo existente."
        assert self.dataset_file.split(".")[-1] == "csv", "`dataset_file` debe ser un archivo CSV."
        
        assert os.path.exists(self.result_dir), "`result_dir` debe ser un directorio existente."

        assert 0.0 < self.train_split and self.train_split < 1.0, "`train_split` debe estar en el rango (0,1)."
