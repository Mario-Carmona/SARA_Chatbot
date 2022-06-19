"""! @brief Definición de argumentos de configuración relacionados con la división de datasets."""

##
# @file split_dataset_arguments.py
#
# @brief Definición de argumentos de la división de datasets.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con la división de datasets para el entrenamiento.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
# - Librería estándar os (https://docs.python.org/3/library/os.html)
# - Librería dataclass.project_arguments
#   - Acceso a la clase ProyectArguments
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import os
from dataclasses import dataclass, field

from dataclass.project_arguments import ProyectArguments


@dataclass
class SplitDatasetArguments(ProyectArguments):
    """! Dataclass para definir los argumentos de la división de datasets.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con la división de datasets para el entrenamiento.
    """

    split_dataset_file: str = field(
        metadata={
            "help": "Nombre del dataset a dividir"
        }
    )
    train_split: float = field(
        metadata={
            "help": "Porcentaje de la división que se convertirá en datos de entrenamiento"
        }
    )
    split_result_dir: str = field(
        metadata={
            "help": "Directorio donde guardar los datasets resultantes de la división"
        }
    )
    train_dataset_file: str = field(
        metadata={
            "help": "Nombre del dataset de entrenamiento resultante de la división"
        }
    )
    valid_dataset_file: str = field(
        metadata={
            "help": "Nombre del dataset de validación resultante de la división"
        }
    )
    seed: int = field(
        metadata={
            "help": "Semilla del generador de números aleatorios"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase SplitDatasetArguments.
        """

        # Completar las rutas de todos los argumentos
        self.split_result_dir = os.path.join(self.workdir, self.split_result_dir)
        if self.split_dataset_file != "":
            self.split_dataset_file = os.path.join(self.workdir, self.split_dataset_file)

        # Comprobaciones de los argumentos
            assert os.path.exists(self.split_dataset_file), "`split_dataset_file` debe ser un archivo existente."
            assert self.split_dataset_file.split('.')[-1] == 'csv', "`split_dataset_file` debe ser un archivo CSV"

        assert 0.0 < self.train_split and self.train_split < 1.0, "`train_split` debe estar en el rango (0,1)."

        assert os.path.exists(self.split_result_dir), "`split_result_dir` debe ser un directorio existente."

        assert self.train_dataset_file.split('.')[-1] == 'csv', "`train_dataset_file` debe ser un archivo CSV"

        assert self.valid_dataset_file.split('.')[-1] == 'csv', "`valid_dataset_file` debe ser un archivo CSV"
