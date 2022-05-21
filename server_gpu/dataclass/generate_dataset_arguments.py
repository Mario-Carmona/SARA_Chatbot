
import os
from dataclasses import dataclass, field

from dataclass.attitude_dataset_arguments import AttitudeDatasetArguments
from dataclass.join_datasets_arguments import JoinDatasetsArguments
from dataclass.split_dataset_arguments import SplitDatasetArguments
from dataclass.theme_dataset_arguments import ThemeDatasetArguments


@dataclass
class GenerateDatasetArguments(JoinDatasetsArguments,
                               SplitDatasetArguments, ThemeDatasetArguments):
    """
    Argumentos relacionados con la generación de datasets para el entrenamiento
    """



    def __post_init__(self):
        JoinDatasetsArguments.__post_init__(self)
        SplitDatasetArguments.__post_init__(self)
        ThemeDatasetArguments.__post_init__(self)

        self.list_datasets.append(self.attitude_dataset_file)

