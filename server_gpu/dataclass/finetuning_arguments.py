"""! @brief Definición de argumentos de configuración relacionados con el proceso de finetuning."""

##
# @file finetuning_arguments.py
#
# @brief Definición de argumentos del finetuning.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con el proceso de finetuning.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
# - Librería estándar os (https://docs.python.org/3/library/os.html)
# - Librería dataclass.model_conversation_arguments
#   - Acceso a la clase ModelConverArguments
# - Librería dataclass.generate_arguments
#   - Acceso a la clase GenerateArguments
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import os
from dataclasses import dataclass, field

from dataclass.model_conversation_arguments import ModelConverArguments
from dataclass.generate_arguments import GenerateArguments


@dataclass
class FinetuningArguments(ModelConverArguments, GenerateArguments):
    """! Dataclass para definir los argumentos del finetuning.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con el proceso de finetuning.
    """
    
    data_dir: str = field(
        metadata={
            "help": "Directorio de los datos de entrada"
        }
    )
    train_dataset: str = field(
        metadata={
            "help": "Nombre del dataset de entrenamiento"
        }
    )
    validation_dataset: str = field(
        metadata={
            "help": "Nombre del dataset de validación"
        }
    )
    task: str = field(
        metadata={
            "help": "Nombre de la tarea que realizará el chatbot"
        }
    )
    max_source_length: int = field(
        metadata={
            "help": "Longitud máxima de las secuencias de las frases fuente tras ser divididas en tokens"
        }
    )
    max_target_length: int = field(
        metadata={
            "help": "Longitud máxima de las secuencias de las frases destino tras ser divididas en tokens"
        }
    )
    n_train: int = field(
        default=-1, 
        metadata={
            "help": "Número de ejemplos de entrenamiento, -1 significa el uso de todos los ejemplos"
        }
    )
    n_val: int = field(
        default=-1, 
        metadata={
            "help": "Número de ejemplos de validación, -1 significa el uso de todos los ejemplos"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase FinetuningArguments.
        """

        # Postprocesdado de la clase ModelConverArguments
        ModelConverArguments.__post_init__(self)

        # Completar las rutas de todos los argumentos
        self.data_dir = os.path.join(self.workdir, self.data_dir)
        self.train_dataset = os.path.join(self.data_dir, self.train_dataset)
        self.validation_dataset = os.path.join(self.data_dir, self.validation_dataset)

        # Comprobaciones de los argumentos
        assert os.path.exists(self.data_dir), "`data_dir` debe ser un directorio existente."

        assert os.path.exists(self.train_dataset), "`train_dataset` debe ser un archivo existente."
        assert self.train_dataset.split(".")[-1] == "csv", "`train_dataset` debe ser un archivo CSV."

        assert os.path.exists(self.validation_dataset), "`validation_dataset` debe ser un archivo existente."
        assert self.validation_dataset.split(".")[-1] == "csv", "`validation_dataset` debe ser un archivo CSV."

        assert self.max_source_length > 0, "`max_source_length` debe ser un entero positivo."

        assert self.max_target_length > 0, "`max_target_length` debe ser un entero positivo."
