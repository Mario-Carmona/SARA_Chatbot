"""! @brief Definición de argumentos de configuración relacionados con la generación de respuestas del servidor."""

##
# @file generate_server_arguments.py
#
# @brief Definición de argumentos de la generación de respuestas.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con la generación de respuestas del servidor.
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
class GenerateServerArguments:
    """! Dataclass para definir los argumentos de la generación de respuestas.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con la generación de respuestas del servidor.
    """

    do_sample: bool = field(
        metadata={
            "help": "Utilizar o no el muestreo mediante técnicas Greedy de decodificación"
        }
    )
    temperature: float = field(
        metadata={
            "help": "El valor utilizado para modular las probabilidades del siguiente token"
        }
    )
    top_p: float = field(
        metadata={
            "help": "El valor que indica el límite mínimo de probabilidades a partir del cuál se conservan los tokens para la generación"
        }
    )
    max_time: float = field(
        metadata={
            "help": "La cantidad máxima de tiempo en segundos que permite que se ejecute el cálculo"
        }
    )
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
    use_cache: bool = field(
        metadata={
            "help": "Uso o no de las últimas atenciones clave/valor para acelerar la decodificación"
        }
    )
