"""! @brief Definición de argumentos de configuración relacionados con la creación del dataset temático."""

##
# @file theme_dataset_arguments.py
#
# @brief Definición de argumentos de la creación del dataset temático.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con la creación del dataset temático.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
# - Librería estándar os (https://docs.python.org/3/library/os.html)
# - Librería dataclass.model_summary_arguments
#   - Acceso a la clase ModelSumArguments
# - Librería dataclass.model_simplify_arguments
#   - Acceso a la clase ModelSimplifyArguments
# - Librería dataclass.deepl_arguments
#   - Acceso a la clase DeeplArguments
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import os
from dataclasses import dataclass, field

from dataclass.model_summary_arguments import ModelSumArguments
from dataclass.model_simplify_arguments import ModelSimplifyArguments
from dataclass.deepl_arguments import DeeplArguments


@dataclass
class ThemeDatasetArguments(ModelSumArguments, ModelSimplifyArguments, 
                            DeeplArguments):
    """! Dataclass para definir los argumentos de la creación del dataset temático.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con la creación del dataset temático.
    """

    seed: int = field(
        metadata={
            "help": "Semilla del generador de números aleatorios"
        }
    )
    max_length_summary: int = field(
        metadata={
            "help": "Máxima longitud en tokens de las frases resumidas"
        }
    )
    num_beams_summary: int = field(
        metadata={
            "help": "Número de haces utilizados para la búsqueda de haces de grupo"
        }
    )
    initial_dataset_file: str = field(
        metadata={
            "help": "Archivo CSV que contiene el dataset inicial para la sección temática"
        }
    )
    translated: bool = field(
        metadata={
            "help": "Indica si `initial_dataset_file` es un archivo traducido o no"
        }
    )
    theme_result_dir: str = field(
        metadata={
            "help": "Directorio que contiene los datasets temáticos"
        }
    )
    adult_dataset_file: str = field(
        metadata={
            "help": "Archivo CSV que contiene el dataset temático para adultos"
        }
    )
    child_dataset_file: str = field(
        metadata={
            "help": "Archivo CSV que contiene el dataset temático para niños"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase ThemeDatasetArguments.
        """
        
        # Postprocesdado de la clase ModelSumArguments
        ModelSumArguments.__post_init__(self)
        # Postprocesdado de la clase ModelSimplifyArguments
        ModelSimplifyArguments.__post_init__(self)
        
        # Completar las rutas de todos los argumentos
        self.initial_dataset_file = os.path.join(self.workdir, self.initial_dataset_file)
        self.theme_result_dir = os.path.join(self.workdir, self.theme_result_dir)

        # Comprobaciones de los argumentos
        assert self.limit_summary >= 0, "`limit_summary` debe ser un entero positivo."

        assert self.max_length_question >= 0, "`max_length_question` debe ser un entero positivo."

        assert self.num_beams_summary >= 0, "`num_beams_summary` debe ser un entero positivo."

        assert self.num_beams_question >= 0, "`num_beams_question` debe ser un entero positivo."

        assert os.path.exists(self.initial_dataset_file), "`initial_dataset_file` debe ser un archivo existente."
        assert self.initial_dataset_file.split('.')[-1] == 'csv', "`initial_dataset_file` debe ser un archivo CSV"

        assert os.path.exists(self.theme_result_dir), "`theme_result_dir` debe ser un directorio existente."

        assert self.adult_dataset_file.split('.')[-1] == 'csv', "`adult_dataset_file` debe ser un archivo CSV"
        
        assert self.child_dataset_file.split('.')[-1] == 'csv', "`child_dataset_file` debe ser un archivo CSV"
