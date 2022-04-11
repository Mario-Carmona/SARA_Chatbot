
import os
from dataclasses import dataclass, field


@dataclass
class GenerateDatasetArguments:
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
    dataset_type: str = field(
        metadata={
            "help": "Tipo de generación del dataset"
        }
    )

    def __post_init__(self):
        assert self.limit_summary >= 0, "`limit_summary` debe ser un entero positivo."

        assert self.max_length_question >= 0, "`max_length_question` debe ser un entero positivo."

        assert self.num_beams_summary >= 0, "`num_beams_summary` debe ser un entero positivo."

        assert self.num_beams_question >= 0, "`num_beams_question` debe ser un entero positivo."

        assert os.path.exists(self.dataset_file), "`dataset_file` debe ser un archivo existente."
        assert self.dataset_file.split(".")[-1] == "csv", "`dataset_file` debe ser un archivo CSV."
        
        assert os.path.exists(self.result_dir), "`result_dir` debe ser un directorio existente."

        assert 0.0 < self.train_split and self.train_split < 1.0, "`train_split` debe estar en el rango (0,1)."

        assert self.dataset_type in ["Niño", "Adulto"], "`dataset_type` debe tener alguno de los siguientes valores: 'Niño', 'Adulto'."
