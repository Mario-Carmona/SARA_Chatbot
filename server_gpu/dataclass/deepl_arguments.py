

"""! @brief Definición de argumentos de configuración relacionados con DeepL."""

##
# @file deepl_arguments.py
#
# @brief Definición de argumentos de DeepL.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con DeepL.
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
class DeeplArguments:
    """! Dataclass para definir los argumentos de DeepL.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con DeepL.
    """
    
    auth_key_deepl: str = field(
        metadata={
            "help": "Clave de autenticación para la API de DeepL"
        }
    )
