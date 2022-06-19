"""! @brief Definición de argumentos de configuración relacionados con el modelo de resumen de frases."""

##
# @file model_summary_arguments.py
#
# @brief Definición de argumentos del modelo de resumen de frases.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con el modelo de resumen de frases.
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
class ModelSumArguments(ProyectArguments):
    """! Dataclass para definir los argumentos del modelo de resumen de frases.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con el modelo de resumen de frases.
    """

    model_summary: str = field(
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_summary_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_summary_tokenizer: str = field(
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_summary_tokenizer_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase ModelSumArguments.
        """

        # Completar las rutas de todos los argumentos
        self.model_summary = os.path.join(self.workdir, self.model_summary)
        self.model_summary_config = os.path.join(self.workdir, self.model_summary_config)
        self.model_summary_tokenizer = os.path.join(self.workdir, self.model_summary_tokenizer)
        self.model_summary_tokenizer_config = os.path.join(self.workdir, self.model_summary_tokenizer_config)

        # Comprobaciones de los argumentos
        assert os.path.exists(self.model_summary), "`model_summary` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_config), "`model_summary_config` debe ser un archivo existente."
        assert os.path.exists(self.model_summary_tokenizer), "`model_summary_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_tokenizer_config), "`model_summary_tokenizer_config` debe ser un archivo existente."
