#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la ejecución de la APP."""


##
# @mainpage Sara Chatbot Heroku APP
#
# @section description_main Descripción
# Programas Python para la ejecución de la APP del sistema Sara Chatbot 
# en la plataforma Heroku.
#
# Copyright (c) 2022.  All rights reserved.


##
# @file main.py
#
# @brief Programa principal para la ejecución de la APP.
#
# @section description_main Descripción
# Programa principal para la ejecución de la APP.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar json (https://docs.python.org/3/library/json.html)
#   - Acceso a la función load
#   - Acceso a la función loads
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función environ
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path.
# - Librería estándar typing (https://docs.python.org/3/library/typing.html)
#   - Acceso al tipo Dict
# - Librería requests 
#   - Acceso a la función post
# - Librería psycopg2 (https://pypi.org/project/psycopg2/)
#   - Acceso a la función connect
# - Librería datetime (https://docs.python.org/3/library/datetime.html)
#   - Acceso a la función datetime
# - Librería pytz (https://pypi.org/project/pytz/)
#   - Acceso a la función timezone
# - Librería estándar enum (https://docs.python.org/3/library/enum.html)
#   - Acceso a la clase Enum
# - Librería pydantic (https://pydantic-docs.helpmanual.io/)
#   - Acceso a la clase BaseModel
# - Librería fastapi (https://fastapi.tiangolo.com/)
#   - Acceso a la función FastAPI
#   - Acceso al dataclass Request
#   - Acceso a la librería fastapi.responses
#       - Acceso a la función HTMLResponse
#       - Acceso a la función PlainTextResponse
#   - Acceso a la librería fastapi.staticfiles
#       - Acceso a la función StaticFiles
#   - Acceso a la librería fastapi.templating
#       - Acceso a la función Jinja2Templates
# - Librería uvicorn (https://www.uvicorn.org/)
#   - Acceso a la función run
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

#   General
import json
import os
from pathlib import Path
from typing import Dict
import enum

#   Gestión peticiones
import requests
from pydantic import BaseModel

#   Gestión del tiempo
import pytz
from datetime import datetime

#   Gestión del log
import psycopg2

#   Despliegue servidor
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn



# Enumerados

class TalkType(enum.Enum):
    """! Enumerado de los tipos de generadores de respuestas.
    Define el enumerado utilizado para distinguir los distintos tipos de generadores de conversaciones.
    """

    child = "child"
    adult = "adult"



# Dataclass

class ServerURL(BaseModel):
    """! Dataclass para recibir URL.
    Define la dataclass utilizada para recibir una URL.
    """

    url: str



# Constantes globales
## Ruta base del programa python.
BASE_PATH = str(Path(__file__).resolve().parent)

"""
    Lectura del archivo de configuración
"""
with open(BASE_PATH + "/config.json") as file:
    config = json.load(file)

## Host del servidor.
HOST = os.environ.get("HOST", config["host"])
## Puerto del servidor.
PORT = eval(os.environ.get("PORT", config["port"]))
## URL del servidor GPU.
SERVER_GPU_URL = os.environ.get("SERVER_GPU_URL", config["server_gpu_url"])
## Zona horaria de España.
SPAIN = pytz.timezone('Europe/Madrid')



# Funciones

def obtenerElemContext(outputContexts):
    """! Obtener índice del contexto que contiene la información de la conversación.
    
    @param outputContexts  Lista de contextos recibidos.
    
    @return Índice del contexto a utilizar.
    """
    
    # Obtención del índice del contexto buscado
    elem = [i for i, s in enumerate(outputContexts) if s["name"].__contains__("talk-followup")][0]

    # Si el contexto buscado no tiene el campo de los parámetros, se le crea. Esto pasa cuando es el inicio de la conversación
    if(not "parameters" in outputContexts[elem].keys()):
        outputContexts[elem]["parameters"] = {}

    return elem


def is_first_response(outputContexts):
    """! Comprobar si se trata del contexto de la primera respuesta de la conversación.
    
    @param outputContexts  Lista de contextos recibidos.
    
    @return Afirmación o negación en cuanto a si es la primera respuesta.
    """

    # Obtención del índice del contexto que contiene la información
    elem = obtenerElemContext(outputContexts)
    
    # Comprobamos si el campo de la edad está dentro de los parámetros del contexto obtenido
    if(not "edad" in outputContexts[elem].get("parameters").keys()):
        # Si no se encuentra el campo de la edad se afirma que es la primera respuesta
        return True
    else:
        # En caso contrario, se niega que es la primera respuesta
        return False


def generate_response(entry: str, edad: str, conver_id: str="", last_response: bool=False):
    """! Generar respuesta.
    
    @param entry          Frase de entrada.
    @param edad           Edad del usuario.
    @param conver_id      Identificador de la conversación.
    @param last_response  Indica si es la última repsuesta de la conversación.
    
    @return Respuesta generada en base a la frase de entrada.
    """
    
    # Datos de la petición POST
    query_json = {
        "entry": entry,
        "conver_id": conver_id,
        "last_response": last_response
    }
    # Cabecera de la petición POST
    headers = {'content-type': 'application/json'}
    # Envío de la petición POST y recepción de la respuesta
    output = requests.post(SERVER_GPU_URL + "/" + edad, json=query_json, headers=headers)
    # Decodificación de la respuesta recibida de la petición POST
    output = json.loads(output.content.decode('utf-8'))

    return output


def make_first_response(request: Dict, edad: str):
    """! Generar la primera respuesta del wekhook.
    
    @param request  Datos de la petición al webhook.
    @param edad     Edad del usuario.

    @return Datos de la primera respuesta del webhook.
    """
    
    # Obtención del contexto de la petición
    outputContexts = request.get("queryResult").get("outputContexts")

    # Si está disponible el servidor GPU
    if SERVER_GPU_URL != "":
        # Obtención de la frase de entrada
        entry = request.get("queryResult").get("queryText")

        # Obtención del índice del contexto que contiene la información
        elem = obtenerElemContext(outputContexts)

        # Generación de la respuesta
        output = generate_response(entry, edad)

        # Obtención de la fecha actual
        date_ini = datetime.now(SPAIN).strftime('%Y-%m-%d %H:%M:%S')

        # Creación del nuevo contexto de la respuesta a la petición
        outputContexts[elem]["parameters"]["context"] = {
            "entry": {
                "ES": [output["entry"]["ES"]],
                "EN": [output["entry"]["EN"]]
            },
            "answer": {
                "ES": [output["answer"]["ES"]],
                "EN": [output["answer"]["EN"]]
            }
        }

        outputContexts[elem]["parameters"]["conver_id"] = output["conver_id"]

        outputContexts[elem]["parameters"]["date_ini"] = date_ini

        outputContexts[elem]["parameters"]["edad"] = edad

        # Obtención de la respuesta generada
        answer = output["answer"]["ES"]
    else:
        # Es caso contrario, se indica su no disponibilidad en la respuesta
        outputContexts = []
        answer = "Servidor GPU no disponible"

    # Creación de la respuesta a la petición al webhook
    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }
    
    return response


def make_rest_response(request: Dict):
    """! Generar el resto de respuestas del wekhook.
    
    @param request  Datos de la petición al webhook.

    @return Datos de la respuesta del webhook.
    """
    
    # Obtención del contexto de la petición
    outputContexts = request.get("queryResult").get("outputContexts")

    # Obtención de la frase de entrada
    entry = request.get("queryResult").get("queryText")
    
    # Obtención del índice del contexto que contiene la información
    elem = obtenerElemContext(outputContexts)

    # Obtención de la edad
    edad = outputContexts[elem]["parameters"]["edad"]

    # Obtención del identificador de la conversación
    conver_id = outputContexts[elem]["parameters"]["conver_id"]

    # Generación de la respuesta
    output = generate_response(entry, edad, conver_id)

    # Creación del nuevo contexto de la respuesta a la petición
    outputContexts[elem]["parameters"]["context"]["entry"]["ES"].append(output["entry"]["ES"])
    outputContexts[elem]["parameters"]["context"]["answer"]["ES"].append(output["answer"]["ES"])

    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["entry"]["EN"])
    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["answer"]["EN"])

    outputContexts[elem]["parameters"]["conver_id"] = output["conver_id"]

    # Obtención de la respuesta generada
    answer = output["answer"]["ES"]

    # Creación de la respuesta a la petición al webhook
    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }
    
    return response


def make_response_talk(request: Dict, edad: TalkType):
    """! Generar las respuestas del wekhook al intent Talk.
    
    @param request  Datos de la petición al webhook.
    @param edad     Edad del usuario.

    @return Datos de la respuesta del webhook.
    """
    
    # Obtención del contexto de la petición
    outputContexts = request.get("queryResult").get("outputContexts")

    # Generación de la respuesta según el orden de la respuesta dentro de la conversación
    if is_first_response(outputContexts):
        # Si es la primera respuesta
        return make_first_response(request, str(edad).split('.')[-1])
    else:
        # En caso contrario
        return make_rest_response(request)


def generarContent(context):
    """! Generar una cadena con todo el contenido de la conversación.
    
    @param context  Contexto con el contenido de la conversación.

    @return Cadena con el contenido de la conversación.
    """
    
    # Cadena que contendrá la conversación completa
    content = ""

    print(context["entry"]["ES"])
    print(context["entry"]["EN"])
    print(context["answer"]["ES"])
    print(context["answer"]["EN"])

    i = 0

    # Formación de la conversación dentro de la cadena
    for entry_ES, answer_ES, entry_EN, answer_EN in zip(context["entry"]["ES"], context["answer"]["ES"], context["entry"]["EN"], context["answer"]["EN"]):
        content += f"\n[USER]: {entry_ES}\n"
        content += f"[USER (EN)]: {entry_EN}\n"
        content += f"[BOT (EN)]: {answer_EN}\n"
        content += f"[BOT]: {answer_ES}\n"

        i += 1
        print(f"{i}\n")
    
    return content


def save_conversation(context, edad, date_ini):
    """! Guardar la conversación en el log.
    
    @param context   Contexto con el contenido de la conversación.
    @param edad      Edad del usuario.
    @param date_ini  Fecha del inicio de la conversación
    """
    
    # Obtención de la URL del log
    DATABASE_URL = os.environ['DATABASE_URL']

    # Abrir conexión con el log
    db = psycopg2.connect(DATABASE_URL, sslmode='require')

    # Obtener cursor del log
    cur = db.cursor()

    # Generar cadena con el contenido de la conversación
    content = generarContent(context)

    # Obtención de la fecha de cierre de la conversación
    date_fin = datetime.now(SPAIN).strftime('%Y-%m-%d %H:%M:%S')

    # Inserción de la conversación dentro del log
    cur.execute(
        """INSERT INTO Conversations (edad, date_ini, date_fin, content) 
            VALUES (
                %s,
                %s,
                %s, 
                %s
            )
        """,
        (
            edad, 
            date_ini,
            date_fin,
            content
        )
    )

    # Indicación de los cambios realizados en el log
    db.commit()

    # Cierre del cursor
    cur.close()

    # Cierre de la conexión del log
    db.close()


def make_response_goodbye(request: Dict):
    """! Generar las respuestas del wekhook al intent Goodbye.
    
    @param request  Datos de la petición al webhook.

    @return Datos de la respuesta del webhook.
    """
    
    
    print("\nGoodbye\n")

    # Obtención del contexto de la petición
    outputContexts = request.get("queryResult").get("outputContexts")

    # Obtención de la frase de entrada
    entry = request.get("queryResult").get("queryText")
    
    # Obtención del índice del contexto que contiene la información
    elem = obtenerElemContext(outputContexts)

    # Obtención de la edad
    edad = outputContexts[elem]["parameters"]["edad"]

    # Obtención del identificador de la conversación
    conver_id = outputContexts[elem]["parameters"]["conver_id"]

    # Generación de la respuesta
    generate_response(entry, edad, conver_id, True)

    # Creación de la respuesta a la petición al webhook
    response = {
        "output_contexts": []
    }

    print("--------")
    print(outputContexts[elem]["parameters"]["context"])
    print("--------")

    # Salvar la conversación en el log
    save_conversation(
        outputContexts[elem]["parameters"]["context"],
        outputContexts[elem]["parameters"]["edad"], 
        outputContexts[elem]["parameters"]["date_ini"]
    )

    return response


def main():
    """! Entrada al programa principal."""

    # Creación de la APP de FastAPI
    app = FastAPI(version="1.0.0")

    # Montaje de los archivos de la carpeta static
    app.mount("/static", StaticFiles(directory=(BASE_PATH + "/static")))
    
    # Crear un objeto templates para el manejo de los HTML
    templates = Jinja2Templates(directory=(BASE_PATH + "/templates"))

    # Ruta raíz
    @app.get("/", response_class=HTMLResponse) 
    def home(request: Request, dark_mode: str = ""):
        """! Función asociada a la ruta raíz.
    
        @param request    Datos de la petición al webhook.
        @param dark_mode  Estado del switch de cambio al modo oscuro

        @return HTML asociado a la página principal.
        """

        return templates.TemplateResponse(
            "home.html", 
            {
                "request": request,
                "dark_mode": dark_mode
            }
        )

    # Ruta a la actividad del chatbot
    @app.get("/chatbot", response_class=HTMLResponse)
    def chatbot(request: Request, dark_mode: str = ""):
        """! Función asociada a la actividad del chatbot.
    
        @param request    Datos de la petición al webhook.
        @param dark_mode  Estado del switch de cambio al modo oscuro

        @return HTML asociado a la actividad del chatbot.
        """

        return templates.TemplateResponse(
            "chatbot.html", 
            {
                "request": request,
                "dark_mode": dark_mode,
                "server_gpu_url": SERVER_GPU_URL,
                "web_interface_adult": config["web_interface_adult"],
                "web_interface_child": config["web_interface_child"],
                "telegram_interface_adult": config["telegram_interface_adult"],
                "telegram_interface_child": config["telegram_interface_child"]
            }
        )

    # Ruta a la sección de reconocimiento de la edad
    @app.get("/capture_image", response_class=HTMLResponse) 
    def capture_image(request: Request, canal: str, dark_mode: str = ""):
        """! Función asociada a la sección de reconocimiento de la edad.
    
        @param request    Datos de la petición al webhook.
        @param canal      Indicador sobre el canal donde se realiza la conversación (Web o Telegram)
        @param dark_mode  Estado del switch de cambio al modo oscuro

        @return HTML asociado a la sección de reconocimiento de la edad.
        """
        
        return templates.TemplateResponse(
            "capture_image.html", 
            {
                "request": request, 
                "canal": canal,
                "dark_mode": dark_mode,
                "server_gpu_url": SERVER_GPU_URL,
                "apartado_deduct": config["apartado_deduct"],
                "web_interface_adult": config["web_interface_adult"],
                "web_interface_child": config["web_interface_child"],
                "telegram_interface_adult": config["telegram_interface_adult"],
                "telegram_interface_child": config["telegram_interface_child"]
            }
        )

    # Ruta a la interfaz del chatbot para adultos
    @app.get("/interface_adult", response_class=HTMLResponse)
    def interface_adult(request: Request, dark_mode: str = ""):
        """! Función asociada a la interfaz del chatbot para adultos.
    
        @param request    Datos de la petición al webhook.
        @param dark_mode  Estado del switch de cambio al modo oscuro

        @return HTML asociado a la interfaz del chatbot para adultos.
        """
        
        return templates.TemplateResponse(
            "interface_adult.html", 
            {
                "request": request,
                "dark_mode": dark_mode
            }
        )

    # Ruta a la interfaz del chatbot para niños
    @app.get("/interface_child", response_class=HTMLResponse)
    def interface_child(request: Request, dark_mode: str = ""):
        """! Función asociada a la interfaz del chatbot para niños.
    
        @param request    Datos de la petición al webhook.
        @param dark_mode  Estado del switch de cambio al modo oscuro

        @return HTML asociado a la interfaz del chatbot para niños.
        """
        
        return templates.TemplateResponse(
            "interface_child.html", 
            {
                "request": request,
                "dark_mode": dark_mode
            }
        )

    # Ruta a la sección de reactivación del servidor
    @app.get("/wakeup", response_class=PlainTextResponse)
    def wakeup():
        """! Función asociada a la sección de reactivación del servidor.

        @return Mensaje indicando el estado activo del servidor.
        """

        return "Server ON"

    # Ruta a la sección de actualización de la URL del servidor GPU
    @app.post("/setURL", response_class=PlainTextResponse)
    def setURL(request: ServerURL):
        """! Función asociada a la sección de actualización de la URL del servidor GPU.

        @param request  URL del servidor GPU.

        @return Mensaje indicando la actualización de la URL del servidor GPU.
        """
        
        # Actualización de la URL del servidor GPU
        global SERVER_GPU_URL
        SERVER_GPU_URL = request.url

        return "URL fijada correctamente"

    # Ruta a la sección de gestión de los mensajes al chatbot para adultos
    @app.post("/webhook_adult")
    async def webhook_adult( request: Request):
        """! Función asociada a la sección de gestión de los mensajes al chatbot para adultos.
    
        @param request  Datos de la petición al webhook.

        @return Respuesta a la petición al webhook.
        """
        
        # Obtención de los datos de la petición en formato JSON
        request_JSON = await request.json()

        print(request_JSON)

        # Obtención del intent que ha enviado a la petición
        intent = request_JSON["queryResult"]["intent"]["displayName"]

        # Actuación del webhook dependiendo del intent que envió la petición
        if intent == "Talk":
            response = make_response_talk(request_JSON, TalkType.adult)
        elif intent == "Goodbye":
            response = make_response_goodbye(request_JSON)

        print(response)

        return response

    # Ruta a la sección de gestión de los mensajes al chatbot para niños
    @app.post("/webhook_child")
    async def webhook_child( request: Request):
        """! Función asociada a la sección de gestión de los mensajes al chatbot para niños.
    
        @param request  Datos de la petición al webhook.

        @return Respuesta a la petición al webhook.
        """

        # Obtención de los datos de la petición en formato JSON
        request_JSON = await request.json()

        print(request_JSON)

        # Obtención del intent que ha enviado a la petición
        intent = request_JSON["queryResult"]["intent"]["displayName"]

        # Actuación del webhook dependiendo del intent que envió la petición
        if intent == "Talk":
            response = make_response_talk(request_JSON, TalkType.child)
        elif intent == "Goodbye":
            response = make_response_goodbye(request_JSON)

        print(response)

        return response


    # Inicio del servidor de la APP
    uvicorn.run(app, host=HOST, port=PORT)



if __name__ == "__main__":
    main()
