
import os
from typing import List
from dataclasses import dataclass, field

from dataclass.model_summary_arguments import ModelSumArguments
from dataclass.model_simplify_arguments import ModelSimplifyArguments
from dataclass.deepl_arguments import DeeplArguments
from dataclass.model_genQuestion_arguments import ModelGenQuestionArguments


@dataclass
class ThemeDatasetArguments(ModelSumArguments, ModelGenQuestionArguments,
                            ModelSimplifyArguments, DeeplArguments):
    """
    Argumentos relacionados con la extracción de ejemplos del dataset `empathetic_dialogues`
    """

    seed: int = field(
        metadata={
            "help": ""
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
    initial_dataset_file: str = field(
        metadata={
            "help": "Archivo CSV que contiene el dataset inicial para la sección temática"
        }
    )
    translated: bool = field(
        metadata={
            "help": "Indica si `initial_dataset_file` es un archivo traducido"
        }
    )
    theme_result_dir: str = field(
        metadata={
            "help": "Archivo CSV que contiene el dataset"
        }
    )
    adult_dataset_file: str = field(
        metadata={
            "help": "Directorio destino de los archivos generados"
        }
    )
    child_dataset_file: str = field(
        metadata={
            "help": "Directorio destino de los archivos generados"
        }
    )

    def __post_init__(self):
        ModelSumArguments.__post_init__(self)
        ModelGenQuestionArguments.__post_init__(self)
        ModelSimplifyArguments.__post_init__(self)
        
        self.initial_dataset_file = os.path.join(self.workdir, self.initial_dataset_file)

        assert self.limit_summary >= 0, "`limit_summary` debe ser un entero positivo."

        assert self.max_length_question >= 0, "`max_length_question` debe ser un entero positivo."

        assert self.num_beams_summary >= 0, "`num_beams_summary` debe ser un entero positivo."

        assert self.num_beams_question >= 0, "`num_beams_question` debe ser un entero positivo."

        assert os.path.exists(self.initial_dataset_file), "`initial_dataset_file` debe ser un archivo existente."
        assert self.initial_dataset_file.aplit('.')[-1] == 'csv', "`initial_dataset_file` debe ser un archivo CSV"

        assert os.path.exists(self.theme_result_dir), "`theme_result_dir` debe ser un directorio existente."

        assert self.adult_dataset_file.aplit('.')[-1] == 'csv', "`adult_dataset_file` debe ser un archivo CSV"
        
        assert self.child_dataset_file.aplit('.')[-1] == 'csv', "`child_dataset_file` debe ser un archivo CSV"
