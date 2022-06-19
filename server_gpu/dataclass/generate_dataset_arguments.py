"""! @brief Definición de argumentos de configuración relacionados con la generación de datasets."""

##
# @file generate_dataset_arguments.py
#
# @brief Definición de argumentos de la generación de datasets.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con la generación de datasets para el entrenamiento.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
# - Librería estándar os (https://docs.python.org/3/library/os.html)
# - Librería dataclass.join_datasets_arguments
#   - Acceso a la clase JoinDatasetsArguments
# - Librería dataclass.split_dataset_arguments
#   - Acceso a la clase SplitDatasetArguments
# - Librería dataclass.theme_dataset_arguments
#   - Acceso a la clase ThemeDatasetArguments
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

from dataclasses import dataclass

from dataclass.join_datasets_arguments import JoinDatasetsArguments
from dataclass.split_dataset_arguments import SplitDatasetArguments
from dataclass.theme_dataset_arguments import ThemeDatasetArguments


@dataclass
class GenerateDatasetArguments(JoinDatasetsArguments,
                               SplitDatasetArguments, ThemeDatasetArguments):
    """! Dataclass para definir los argumentos de la generación de datasets.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con la generación de datasets para el entrenamiento.
    """

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase GenerateDatasetArguments.
        """

        # Postprocesdado de la clase JoinDatasetsArguments
        JoinDatasetsArguments.__post_init__(self)
        # Postprocesdado de la clase SplitDatasetArguments
        SplitDatasetArguments.__post_init__(self)
        # Postprocesdado de la clase ThemeDatasetArguments
        ThemeDatasetArguments.__post_init__(self)
