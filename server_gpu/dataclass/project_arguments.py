"""! @brief Definición de argumentos de configuración relacionados con el proyecto."""

##
# @file project_arguments.py
#
# @brief Definición de argumentos del proyecto.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con el proyecto.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import os
from dataclasses import dataclass, field


@dataclass
class ProyectArguments:
    """! Dataclass para definir los argumentos del proyecto.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con el proyecto.
    """

    workdir: str = field(
        metadata={
            "help": "Directorio de trabajo del proyecto"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase ProyectArguments.
        """

        # Comprobaciones de los argumentos
        assert os.path.exists(self.workdir), "`workdir` debe ser un directorio existente."
