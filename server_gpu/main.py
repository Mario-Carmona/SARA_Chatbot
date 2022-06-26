#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la ejecución del servidor GPU."""


##
# @mainpage Servidor GPU
#
# @section description_main Descripción
# Programas Python para la ejecución del servidor GPU mediante FastAPI y ngrok.
#
# Copyright (c) 2022.  All rights reserved.


##
# @file main.py
#
# @brief Programa principal para la ejecución del servidor GPU.
#
# @section description_main Descripción
# Programa principal para la ejecución del servidor GPU.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función environ
#   - Acceso a la función getenv
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path
# - Librería color
#   - Acceso a la clase bcolors
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería dataclass.server_arguments
#   - Acceso a la clase ServerArguments
# - Librería transformers (https://pypi.org/project/transformers)
#   - Acceso a la función HfArgumentParser
#   - Acceso a la función set_seed
#   - Acceso a la función AutoConfig
#   - Acceso a la función AutoTokenizer
#   - Acceso a la función BlenderbotForConditionalGeneration
#   - Acceso a la función ConversationalPipeline
#   - Acceso a la función Conversation
#   - Acceso a la función ViTFeatureExtractor
#   - Acceso a la función ViTForImageClassification
# - Librería fastapi (https://fastapi.tiangolo.com/)
#   - Acceso a la función FastAPI
#   - Acceso a la librería fastapi.responses
#       - Acceso a la función PlainTextResponse
#   - Acceso a la librería fastapi.middleware.cors
#       - Acceso a la función CORSMiddleware
# - Librería uvicorn (https://www.uvicorn.org/)
#   - Acceso a la función run
# - Librería pyngrok (https://pyngrok.readthedocs.io/en/latest/)
#   - Acceso a la función ngrok
#       - Acceso a la función connect
#   - Acceso a la función conf
#       - Acceso a la función PyngrokConfig
# - Librería requests 
#   - Acceso a la función post
# - Librería pydantic (https://pydantic-docs.helpmanual.io/)
#   - Acceso a la clase BaseModel
# - Librería torch (https://pypi.org/project/torch/) 
#   - Acceso a la función cuda.set_device
#   - Acceso a la función distributed.get_rank
# - Librería deepl (https://www.deepl.com/es/docs-api/) 
#   - Acceso a la función Translator
# - Librería estándar base64 (https://docs.python.org/3/library/base64.html)
#   - Acceso a la función b64decode
# - Librería PIL (https://pillow.readthedocs.io/en/stable/)
#   - Acceso a la función Image
#       - Acceso a la función open
# - Librería estándar io (https://docs.python.org/3/library/io.html)
#   - Acceso a la función BytesIO
# - Librería deepspeed (https://deepspeed.readthedocs.io/en/latest/)
#   - Acceso a la función init_distributed
#   - Acceso a la función init_inference
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

#   General
import os
from pathlib import Path
from color import bcolors

#   Configuración
import argparse
from dataclass.server_arguments import ServerArguments
from transformers import HfArgumentParser

#   Despliegue servidor
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok, conf

#   Gestión peticiones
import requests
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

#   Modelos
import torch
from transformers import set_seed
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    ConversationalPipeline, 
    Conversation,
    ViTFeatureExtractor,
    ViTForImageClassification
)

#   Módulo de traducción
import deepl

#   Manejo de imágenes
import base64 
from PIL import Image
import io

#   Optimizadores de los modelos
import deepspeed



# Dataclass

class Entry(BaseModel):
    """! Dataclass para recibir una petición de generación de respuesta.
    Define la dataclass utilizada para recibir una petición de generación de respuesta.
    """

    entry: str
    conver_id: str
    last_response: bool

class EntryDeduct(BaseModel):
    """! Dataclass para recibir una petición de deducción de edad.
    Define la dataclass utilizada para recibir una petición de deducción de edad.
    """

    imagen: str










# Analizador de argumentos
parser = argparse.ArgumentParser()

# Añadir un argumento para el archivo de configuración
parser.add_argument(
    "--config_file", 
    type = str,
    help = "El formato del archivo debe ser \'config.json\'"
)

# Añadir un argumento para el local_rank
parser.add_argument(
    "--local_rank", 
    type = int,
    help = "El formato del archivo debe ser \'config.json\'"
)

# Obtención de los argumentos
args = parser.parse_args()



# Constantes globales
## Ruta base del programa python.
BASE_PATH = Path(__file__).resolve().parent

## Archivo de configuración
CONFIG_FILE = args.config_file

# Analizador de argumentos de la librería transformers
parser = HfArgumentParser(
    (
        ServerArguments
    )
)

# Obtención de los argumentos del servidor
server_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

## Espacio de trabajo del servidor
WORKDIR = server_args.workdir



# Fijar ruta donde instalar las extensiones de Pytorch
os.environ["TORCH_EXTENSIONS_DIR"] = WORKDIR + "torch_extensions"

# Fijar como desactivado el paralelismo al convertir las frases en tokens para evitar problemas
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Obtener el identificador de proceso en la máquina local
local_rank = args.local_rank
# Obtener el número de procesos que están trabajando en el script
world_size = int(os.getenv("WORLD_SIZE", "1"))

# Fijar como dispositivo al proceso obtenido
torch.cuda.set_device(local_rank)

# Iniciar el trabajo distribuido, en caso de que haya varios procesos trabajando a la vez
deepspeed.init_distributed()

# Fijar semilla del generador de números aleatorios
set_seed(0)


# Carga de la configuración del modelo conversacional para adultos
configConverAdult = AutoConfig.from_pretrained(
    server_args.model_conver_adult_config
)

# Carga de la configuración del modelo conversacional para niños
configConverChild = AutoConfig.from_pretrained(
    server_args.model_conver_child_config
)

# Carga del tokenizer del modelo conversacional para adultos
tokenizerConverAdult = AutoTokenizer.from_pretrained(
    server_args.model_conver_adult_tokenizer,
    config=server_args.model_conver_adult_tokenizer_config,
    use_fast=True
)

# Carga del tokenizer del modelo conversacional para niños
tokenizerConverChild = AutoTokenizer.from_pretrained(
    server_args.model_conver_child_tokenizer,
    config=server_args.model_conver_child_tokenizer_config,
    use_fast=True
)

# Carga del modelo conversacional para adultos
modelConverAdult = BlenderbotForConditionalGeneration.from_pretrained(
    server_args.model_conver_adult,
    from_tf=bool(".ckpt" in server_args.model_conver_adult),
    config=configConverAdult
)

# Carga del modelo conversacional para niños
modelConverChild = BlenderbotForConditionalGeneration.from_pretrained(
    server_args.model_conver_child,
    from_tf=bool(".ckpt" in server_args.model_conver_child),
    config=configConverChild
)

# Creación del pipeline conversacional para adultos
pipelineConverAdult = ConversationalPipeline(
    model = modelConverAdult,
    tokenizer = tokenizerConverAdult,
    framework = "pt",
    device = local_rank
)

# Creación del pipeline conversacional para niños
pipelineConverChild = ConversationalPipeline(
    model = modelConverChild,
    tokenizer = tokenizerConverChild,
    framework = "pt",
    device = local_rank
)

# Optimización del pipeline para adultos
pipelineConverAdult.model = deepspeed.init_inference(
    pipelineConverAdult.model,
    mp_size=world_size,
    replace_method='auto',
    replace_with_kernel_inject=True
)

# Optimización del pipeline para niños
pipelineConverChild.model = deepspeed.init_inference(
    pipelineConverChild.model,
    mp_size=world_size,
    replace_method='auto',
    replace_with_kernel_inject=True
)

# Creación del traductor
translator = deepl.Translator(server_args.auth_key_deepl)



# Diccionario que contendrá las conversaciones activas en el servidor
dicc_conversation = {}



# Funciones

def make_response_adult(entry: str, conver_id: str, last_response: bool):
    """! Generar respuesta a una petición de generación de respuestas para adultos.
    
    @param entry          Texto de entrada.
    @param conver_id      Identificador de la conversación
    @param last_response  Indica si es o no el final de la conversación
    
    @return Respuesta a la petición.
    """

    # Tradución al ingles del texto de entrada
    entry_EN = translator.translate_text(entry, source_lang="ES", target_lang="EN-US").text

    print(entry_EN)



    # Obtención de la conversación
    try:
        # En el caso de que exista se extrae del diccionario de conversaciones
        conversation = dicc_conversation[conver_id]
    except KeyError:
        # En caso contrario, se genera la conversación
        conversation = Conversation()

    # Añadir a la conversación el texto de entrada traducido
    conversation.add_user_input(entry_EN)

    # Generación de la conversación de respuesta
    output = pipelineConverAdult(
        conversation,
        do_sample=server_args.do_sample,
        temperature=server_args.temperature,
        top_p=server_args.top_p,
        max_time=server_args.max_time,
        max_length=server_args.max_length,
        min_length=server_args.min_length,
        use_cache=server_args.use_cache,
        pad_token_id=tokenizerConverAdult.eos_token_id,
        synced_gpus=True
    )

    print(output)

    # Actualización del diccionario de conversaciones
    if last_response:
        # Si es el final de la conversación se elimina la misma del diccionario
        del dicc_conversation[str(output.uuid)]
    else:
        # En caso contrario, se actualiza el campo que contiene a la conversación
        dicc_conversation[str(output.uuid)] = output

    # Obtención del texto de respuesta 
    answer_EN = output.generated_responses[-1]

    print(answer_EN)

    # Traducción al español del texto de respuesta
    answer = translator.translate_text(answer_EN, source_lang="EN", target_lang="ES").text

    print(answer)

    # Creación de la respuesta a la petición
    response = {
        "entry": {
            "ES": entry, 
            "EN": entry_EN
        },
        "answer": {
            "ES": answer, 
            "EN": answer_EN
        },
        "conver_id": str(output.uuid)
    }

    return response


def make_response_child(entry: str, conver_id: str, last_response: bool):
    """! Generar respuesta a una petición de generación de respuestas para niños.
    
    @param entry          Texto de entrada.
    @param conver_id      Identificador de la conversación
    @param last_response  Indica si es o no el final de la conversación
    
    @return Respuesta a la petición.
    """

    # Tradución al ingles del texto de entrada
    entry_EN = translator.translate_text(entry, source_lang="ES", target_lang="EN-US").text

    print(entry_EN)


    # Obtención de la conversación
    try:
        # En el caso de que exista se extrae del diccionario de conversaciones
        conversation = dicc_conversation[conver_id]
    except KeyError:
        # En caso contrario, se genera la conversación
        conversation = Conversation()

    # Añadir a la conversación el texto de entrada traducido
    conversation.add_user_input(entry_EN)

    # Generación de la conversación de respuesta
    output = pipelineConverChild(
        conversation,
        do_sample=server_args.do_sample,
        temperature=server_args.temperature,
        top_p=server_args.top_p,
        max_time=server_args.max_time,
        max_length=server_args.max_length,
        min_length=server_args.min_length,
        use_cache=server_args.use_cache,
        pad_token_id=tokenizerConverAdult.eos_token_id,
        synced_gpus=True
    )

    print(output)

    # Actualización del diccionario de conversaciones
    if last_response:
        # Si es el final de la conversación se elimina la misma del diccionario
        del dicc_conversation[str(output.uuid)]
    else:
        # En caso contrario, se actualiza el campo que contiene a la conversación
        dicc_conversation[str(output.uuid)] = output

    # Obtención del texto de respuesta 
    answer_EN = output.generated_responses[-1]

    print(answer_EN)

    # Traducción al español del texto de respuesta
    answer = translator.translate_text(answer_EN, source_lang="EN", target_lang="ES").text

    print(answer)

    # Creación de la respuesta a la petición
    response = {
        "entry": {
            "ES": entry, 
            "EN": entry_EN
        },
        "answer": {
            "ES": answer, 
            "EN": answer_EN
        },
        "conver_id": str(output.uuid)
    }

    return response


def send_public_URL():
    """! Envío de la URL pública del servidor al Controlador."""

    print(bcolors.WARNING + "Enviando URL al controlador..." + bcolors.RESET)

    # Obtención de la URL pública del servidor
    url = server_args.controller_url

    # Cabecera de la petición
    headers = {'content-type': 'application/json'}

    # Envío de la petición POST y obtención de la respuesta a la petición
    response = requests.post(url + "/setURL", json={"url": public_url}, headers=headers)
    
    print(bcolors.OK + "INFO" + bcolors.RESET + ": " + str(response.content.decode('utf-8')))


def main():
    """! Entrada al programa principal."""

    # Creación de la APP de FastAPI
    app = FastAPI(version="1.0.0")

    # Añadir un middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ruta raíz
    @app.get("/", response_class=PlainTextResponse)
    def home():
        """! Función asociada a la ruta raíz.
    
        @return Texto que indica la actividad del servidor.
        """

        return "Server GPU ON"

    # Ruta a la generación de respuestas para adultos
    @app.post("/adult")
    def adult(request: Entry):
        """! Función asociada a la generación de respuestas para adultos.
    
        @param request  Petición de generación de respuesta

        @return Respuesta a la petición.
        """

        # Generación de la respuesta para la petición para adultos
        response = make_response_adult(
            request.entry, 
            request.conver_id,
            request.last_response
        )

        return response

    # Ruta a la generación de respuestas para niños
    @app.post("/child")
    def child(request: Entry):
        """! Función asociada a la generación de respuestas para niños.
    
        @param request  Petición de generación de respuesta

        @return Respuesta a la petición.
        """

        # Generación de la respuesta para la petición para niños
        response = make_response_child(
            request.entry, 
            request.conver_id,
            request.last_response
        )

        return response

    # Ruta a la deducción de edad
    @app.post("/deduct", response_class=PlainTextResponse)
    def deduct(request: EntryDeduct):
        """! Función asociada a la deducción de edad.
    
        @param request  Petición de deducción de edad

        @return Edad deducida.
        """

        # Obtener los datos de la imagen recibida con la petición
        base64_data = request.imagen.split(',')[1]

        # Convertir la imagen en bytes  
        img_bytes = base64.b64decode(base64_data)

        # Convertir la imagen en bytes en un objeto PIL Image
        img = Image.open(io.BytesIO(img_bytes))

        # Creación del modelo clasificador de imágenes
        model = ViTForImageClassification.from_pretrained('/mnt/homeGPU/mcarmona/nateraw/vit-age-classifier')
        # Creación del extractor de características
        transforms = ViTFeatureExtractor.from_pretrained('/mnt/homeGPU/mcarmona/nateraw/vit-age-classifier')

        # Extraer las características de la imagen
        inputs = transforms(img, return_tensors='pt')
        # Clasificación de la imagen
        output = model(**inputs)

        # Obtención de las probabilidades de cada clase predecida
        proba = output.logits.softmax(1)

        print(proba)

        # Cálculo de la clase más probable
        preds = proba.argmax(1)

        """
        Los índices empiezan en 0
        0-2
        3-9
        10-19
        20-29
        30-39
        40-49
        50-59
        60-69
        more than 70
        """

        # Elección del rango de edad a devolver
        if int(preds) <= 1:
            print("child")
            return "child"
        else:
            print("adult")
            return "adult"


    # Ruta a la reconexión con el Controlador
    @app.get("/Reconnect", response_class=PlainTextResponse)
    def reconnect():
        """! Función asociada a la reconexión con el Controlador.

        @return Texto indicando la finalización del proceso de reconexión.
        """

        # Envío de la URL pública del servidor al Controlador
        send_public_URL()

        return "Reenviada URL"


    # Obtener rango del proceso que está ejecutando el script
    rank = torch.distributed.get_rank()

    # Sólo el proceso 0 ejecuta el inicio del servidor
    if rank == 0:
        # Obtención del puerto del servidor
        port = eval(os.environ.get("PORT", server_args.port))

        # Definición de la configuración de ngrok
        pyngrok_config = conf.PyngrokConfig(
            ngrok_path=server_args.ngrok_path,
            config_path=server_args.ngrok_config_path
        )

        # Obtención de la URL pública del servidor al conectar con ngrok
        global public_url
        public_url = ngrok.connect(
            port, 
            pyngrok_config=pyngrok_config
        ).public_url

        # Convertir URL HTTP en HTTPS
        if public_url.split(':')[0] == "http":
            public_url = public_url.split(':')[0] + "s:" + public_url.split(':')[1]

        print(bcolors.OK + "Public URL" + bcolors.RESET + ": " + public_url)

        # Envío de la URL pública del servidor al Controlador
        send_public_URL()

        # killall ngrok  → Para eliminar todas las sessiones de ngrok

        # Inicio del servidor GPU
        uvicorn.run(app, port=port)

        # deepspeed --num_gpus 1 main.py --config_file configs/config_server.json
    


if __name__ == "__main__":
    main()
