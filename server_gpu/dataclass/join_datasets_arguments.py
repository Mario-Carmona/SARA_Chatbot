"""! @brief Definición de argumentos de configuración relacionados con la unión de datasets."""

##
# @file join_datasets_arguments.py
#
# @brief Definición de argumentos de la unión de datasets.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con la unión de datasets.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
# - Librería estándar os (https://docs.python.org/3/library/os.html)
# - Librería estándar typing (https://docs.python.org/3/library/typing.html)
#   - Acceso a la clase List
# - Librería dataclass.project_arguments
#   - Acceso a la clase ProyectArguments
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import os
from typing import List
from dataclasses import dataclass, field

from dataclass.project_arguments import ProyectArguments


@dataclass
class JoinDatasetsArguments(ProyectArguments):
    """! Dataclass para definir los argumentos de la unión de datasets.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con la unión de datasets.
    """

    list_datasets: List[str] = field(
        metadata={
            "help": "Lista de datasets a unir"
        }
    )
    join_dataset_file: str = field(
        metadata={
            "help": "Nombre del dataset resultante de la unión"
        }
    )
    remove_source_files: bool = field(
        metadata={
            "help": "Eliminación o no de los archivos indicados en la lista tras realizar la unión"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase JoinDatasetsArguments.
        """

        # Completar las rutas de todos los argumentos
        self.join_dataset_file = os.path.join(self.workdir, self.join_dataset_file)
        for i, dataset_path in enumerate(self.list_datasets):
            self.list_datasets[i] = os.path.join(self.workdir, dataset_path)

        # Comprobaciones de los argumentos
        assert len(self.list_datasets) > 0, "`list_datasets` debe ser una lista con al menos un elemento"
        for dataset_path in self.list_datasets:
            assert os.path.exists(dataset_path), "`list_datasets` debe contener archivos existentes."
            assert dataset_path.split('.')[-1] == 'csv', "`list_datasets` debe contener archivos CSV"

        dir = '/'.join(self.join_dataset_file.split('/')[:-1])
        assert os.path.exists(dir), "`join_dataset_file` debe estar en un directorio existente."
        assert self.join_dataset_file.split('.')[-1] == 'csv', "`join_dataset_file` debe ser un archivo CSV"
