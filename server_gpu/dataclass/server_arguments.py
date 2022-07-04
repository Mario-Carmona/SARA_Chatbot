"""! @brief Definición de argumentos de configuración relacionados con el servidor."""

##
# @file server_arguments.py
#
# @brief Definición de argumentos del servidor.
#
# @section description_main Descripción
# Definición de argumentos de configuración relacionados con el servidor.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar dataclasses (https://docs.python.org/3/library/dataclasses.html)
#   - Acceso a la función dataclass.
#   - Acceso a la función field.
# - Librería estándar os (https://docs.python.org/3/library/os.html)
# - Librería dataclass.deepl_arguments
#   - Acceso a la clase DeeplArguments
# - Librería dataclass.generate_server_arguments
#   - Acceso a la clase GenerateServerArguments
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

from dataclass.deepl_arguments import DeeplArguments
from dataclass.generate_server_arguments import GenerateServerArguments
from dataclass.project_arguments import ProyectArguments


@dataclass
class ServerArguments(ProyectArguments, DeeplArguments, 
                      GenerateServerArguments):
    """! Dataclass para definir los argumentos del servidor.
    Define la dataclass utilizada para definir los argumentos de configuración relacionados con el servidor.
    """

    host: str = field(
        metadata={
            "help": "Dirección host del servidor GPU"
        }
    )
    port: str = field(
        metadata={
            "help": "Puerto del servidor GPU"
        }
    )
    controller_url: str = field(
        metadata={
            "help": "Dirección URL del servidor del controlador"
        }
    )
    ngrok_path: str = field(
        metadata={
            "help": "Ruta al ejecutable de ngrok"
        }
    )
    ngrok_config_path: str = field(
        metadata={
            "help": "Ruta al archivo de configuración de ngrok"
        }
    )
    model_conver_adult: str = field(
        metadata={
            "help": "Ruta a la carpeta del modelo para adultos"
        }
    )
    model_conver_adult_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del modelo para adultos"
        }
    )
    model_conver_adult_tokenizer: str = field(
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo para adultos"
        }
    )
    model_conver_adult_tokenizer_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo para adultos"
        }
    )
    model_conver_child: str = field(
        metadata={
            "help": "Ruta a la carpeta del modelo para niños"
        }
    )
    model_conver_child_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del modelo para niños"
        }
    )
    model_conver_child_tokenizer: str = field(
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo para niños"
        }
    )
    model_conver_child_tokenizer_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo para niños"
        }
    )
    model_deduct_age: str = field(
        metadata={
            "help": "Ruta a la carpeta del modelo para deducir la edad a partir de imágenes"
        }
    )
    tam_history: int = field(
        metadata={
            "help": "Número máximo de tokens que forman el historial de la conversación"
        }
    )

    def __post_init__(self):
        """! Postprocesado de los argumentos.
        
        @param self  Instancia de la clase ServerArguments.
        """

        # Completar las rutas de todos los argumentos
        self.ngrok_path = os.path.join(self.workdir, self.ngrok_path)
        self.ngrok_config_path = os.path.join(self.workdir, self.ngrok_config_path)

        self.model_conver_adult = os.path.join(self.workdir, self.model_conver_adult)
        self.model_conver_adult_config = os.path.join(self.workdir, self.model_conver_adult_config)
        self.model_conver_adult_tokenizer = os.path.join(self.workdir, self.model_conver_adult_tokenizer)
        self.model_conver_adult_tokenizer_config = os.path.join(self.workdir, self.model_conver_adult_tokenizer_config)

        self.model_conver_child = os.path.join(self.workdir, self.model_conver_child)
        self.model_conver_child_config = os.path.join(self.workdir, self.model_conver_child_config)
        self.model_conver_child_tokenizer = os.path.join(self.workdir, self.model_conver_child_tokenizer)
        self.model_conver_child_tokenizer_config = os.path.join(self.workdir, self.model_conver_child_tokenizer_config)

        # Comprobaciones de los argumentos
        assert os.path.exists(self.ngrok_path), "`ngrok_path` debe ser un archivo existente."

        assert os.path.exists(self.ngrok_config_path), "`ngrok_config_path` debe ser un archivo existente."
        assert self.ngrok_config_path.split(".")[-1] == "yml", "`ngrok_config_path` debe ser un archivo YAML."

        assert os.path.exists(self.model_conver_adult), "`model_conver_adult` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_adult_config), "`model_conver_adult_config` debe ser un archivo existente."
        assert os.path.exists(self.model_conver_adult_tokenizer), "`model_conver_adult_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_adult_tokenizer_config), "`model_conver_adult_tokenizer_config` debe ser un archivo existente."

        assert os.path.exists(self.model_conver_child), "`model_conver_child` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_child_config), "`model_conver_child_config` debe ser un archivo existente."
        assert os.path.exists(self.model_conver_child_tokenizer), "`model_conver_child_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_child_tokenizer_config), "`model_conver_child_tokenizer_config` debe ser un archivo existente."

        assert self.tam_history > 0, "`tam_history` debe ser un entero mayor que 0."
