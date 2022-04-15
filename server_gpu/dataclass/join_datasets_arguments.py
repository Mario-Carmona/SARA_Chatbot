
import os
from typing import List
from dataclasses import dataclass, field
from dataclass.project_arguments import ProyectArguments


@dataclass
class JoinDatasetsArguments(ProyectArguments):
    """
    Argumentos relacionados con la extracción de ejemplos del dataset `empathetic_dialogues`
    """

    list_datasets: List[str] = field(
        metadata={
            "help": ""
        }
    )
    result_file: str = field(
        metadata={
            "help": ""
        }
    )
    remove_source_files: bool = field(
        metadata={
            "help": ""
        }
    )

    def __post_init__(self):
        self.result_file = self.workdir + self.result_file

        assert len(self.list_datasets) > 0, "`list_datasets` debe ser una lista con al menos un elemento"
        for dataset in self.list_datasets:
            assert os.path.exists(dataset), "`list_datasets` debe contener archivos existentes."

        dir = '/'.join(self.result_file.aplit('/')[:-1])
        assert os.path.exists(dir), "`result_file` debe estar en un directorio existente."
        assert self.result_file.aplit('.')[-1] == 'csv', "`result_file` debe ser un archivo CSV"
