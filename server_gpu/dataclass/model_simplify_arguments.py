"""! @brief Definición de argumentos de configuración relacionados con el modelo de simplificación de frases."""

##
# @file model_simplify_arguments.py
#
# @brief Definición de argumentos del modelo de simplificación de frases.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con el modelo de simplificación de frases.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
# - Librería SentenceSimplification.muss.simplify
#   - Acceso a la lista ALLOWED_MODEL_NAMES
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

from dataclasses import dataclass, field
from SentenceSimplification.muss.simplify import ALLOWED_MODEL_NAMES


@dataclass
class ModelSimplifyArguments:
    """! Dataclass para definir los argumentos del modelo de simplificación de frases.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con el modelo de simplificación de frases.
    """
    
    model_simplify: str = field(
        metadata={
            "help": "Modelo preentrenado usado para la simplificación"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase ModelSimplifyArguments.
        """

        # Comprobaciones de los argumentos
        assert self.model_simplify in ALLOWED_MODEL_NAMES, "`model_simplify` debe ser uno de los modelo preentrenados disponibles."
