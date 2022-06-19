"""! @brief Definición de argumentos de configuración relacionados con la generación de respuestas durante el proceso de finetuning."""

##
# @file generate_arguments.py
#
# @brief Definición de argumentos de la generación de respuestas.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con la generación de respuestas durante el proceso de finetuning.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

from dataclasses import dataclass, field


@dataclass
class GenerateArguments:
    """! Dataclass para definir los argumentos de la generación de respuestas.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con la generación de respuestas durante el proceso de finetuning.
    """

    max_length: int = field(
        metadata={
            "help": "La longitud máxima de la secuencia que se va a generar"
        }
    )
    min_length: int = field(
        metadata={
            "help": "La longitud mínima de la secuencia que se va a generar"
        }
    )
