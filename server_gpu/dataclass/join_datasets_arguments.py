
import os
from typing import List
from dataclasses import dataclass, field
from dataclass.project_arguments import ProyectArguments


@dataclass
class JoinDatasetsArguments(ProyectArguments):
    """
    Argumentos relacionados con la unión de datasets
    """

    list_datasets: List[str] = field(
        metadata={
            "help": ""
        }
    )
    join_dataset_file: str = field(
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
        self.join_dataset_file = os.path.join(self.workdir, self.join_dataset_file)
        for i, dataset_path in enumerate(self.list_datasets):
            self.list_datasets[i] = os.path.join(self.workdir, dataset_path)

        assert len(self.list_datasets) > 0, "`list_datasets` debe ser una lista con al menos un elemento"
        for dataset_path in self.list_datasets:
            assert os.path.exists(dataset_path), "`list_datasets` debe contener archivos existentes."
            assert self.dataset_path.split('.')[-1] == 'csv', "`list_datasets` debe contener archivos CSV"

        dir = '/'.join(self.join_dataset_file.split('/')[:-1])
        assert os.path.exists(dir), "`join_dataset_file` debe estar en un directorio existente."
        assert self.join_dataset_file.split('.')[-1] == 'csv', "`join_dataset_file` debe ser un archivo CSV"
