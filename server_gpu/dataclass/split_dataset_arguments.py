
import os
from typing import List
from dataclasses import dataclass, field
from dataclass.project_arguments import ProyectArguments


@dataclass
class SplitDatasetArguments(ProyectArguments):
    """
    Argumentos relacionados con la unión de datasets
    """

    split_dataset_file: str = field(
        metadata={
            "help": ""
        }
    )
    train_split: float = field(
        metadata={
            "help": ""
        }
    )
    split_result_dir: str = field(
        metadata={
            "help": ""
        }
    )
    train_dataset_file: str = field(
        metadata={
            "help": ""
        }
    )
    valid_dataset_file: str = field(
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
        self.split_result_dir = os.path.join(self.workdir, self.split_result_dir)
        if self.split_dataset_file != "":
            self.split_dataset_file = os.path.join(self.workdir, self.split_dataset_file)

            assert os.path.exists(self.split_dataset_file), "`split_dataset_file` debe ser un archivo existente."
            assert self.split_dataset_file.split('.')[-1] == 'csv', "`split_dataset_file` debe ser un archivo CSV"

        assert 0.0 < self.train_split and self.train_split < 1.0, "`train_split` debe estar en el rango (0,1)."

        assert os.path.exists(self.split_result_dir), "`split_result_dir` debe ser un directorio existente."

        assert self.train_dataset_file.split('.')[-1] == 'csv', "`train_dataset_file` debe ser un archivo CSV"

        assert self.valid_dataset_file.split('.')[-1] == 'csv', "`valid_dataset_file` debe ser un archivo CSV"
