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
# @section notes_main Notes
# - Add special project notes here that you want to communicate to the user.
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
# - Librería estándar time (https://docs.python.org/3/library/time.html)
#   - Acceso a la función time.
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
# @section notes_doxygen_example Notes
# - Comments are Doxygen compatible.
#
# @section todo_doxygen_example TODO
# - None.
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import json
from time import time
import os
from pathlib import Path
from typing import Dict
import enum

import requests
import psycopg2
import pytz
from datetime import datetime

from pydantic import BaseModel

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
    query_json = {
        "entry": entry,
        "conver_id": conver_id,
        "last_response": last_response
    }
    headers = {'content-type': 'application/json'}
    output = requests.post(SERVER_GPU_URL + "/" + edad, json=query_json, headers=headers)
    output = json.loads(output.content.decode('utf-8'))

    return output






def make_first_response(request: Dict, edad: str):
    outputContexts = request.get("queryResult").get("outputContexts")

    if SERVER_GPU_URL != "":
        entry = request.get("queryResult").get("queryText")

        elem = obtenerElemContext(outputContexts)

        output = generate_response(entry, edad)

        date_ini = datetime.now(SPAIN).strftime('%Y-%m-%d %H:%M:%S')

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

        answer = output["answer"]["ES"]
    else:
        outputContexts = []
        answer = "Servidor GPU no disponible"

    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }
    
    return response



def make_rest_response(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")
    
    elem = obtenerElemContext(outputContexts)

    edad = outputContexts[elem]["parameters"]["edad"]

    conver_id = outputContexts[elem]["parameters"]["conver_id"]

    output = generate_response(entry, edad, conver_id)

    outputContexts[elem]["parameters"]["context"]["entry"]["ES"].append(output["entry"]["ES"])
    outputContexts[elem]["parameters"]["context"]["answer"]["ES"].append(output["answer"]["ES"])

    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["entry"]["EN"])
    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["answer"]["EN"])

    outputContexts[elem]["parameters"]["conver_id"] = output["conver_id"]

    answer = output["answer"]["ES"]

    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }
    
    return response











def make_response_talk(request: Dict, edad: TalkType):
    outputContexts = request.get("queryResult").get("outputContexts")

    if is_first_response(outputContexts):
        return make_first_response(request, str(edad).split('.')[-1])
    else:
        return make_rest_response(request)





def generarContent(context):
    content = ""

    for entry_ES, answer_ES, entry_EN, answer_EN in zip(context["entry"]["ES"], context["answer"]["ES"], context["entry"]["EN"], context["answer"]["EN"]):
        content += f"\n[USER]: {entry_ES}\n"
        content += f"[USER (EN)]: {entry_EN}\n"
        content += f"[BOT (EN)]: {answer_EN}\n"
        content += f"[BOT]: {answer_ES}\n"
    
    return content

def save_conversation(context, edad, date_ini):
    DATABASE_URL = os.environ['DATABASE_URL']

    db = psycopg2.connect(DATABASE_URL, sslmode='require')

    cur = db.cursor()

    content = generarContent(context)

    date_fin = datetime.now(SPAIN).strftime('%Y-%m-%d %H:%M:%S')

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

    db.commit()

    cur.close()
    db.close()




def make_response_goodbye(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")
    
    elem = obtenerElemContext(outputContexts)

    edad = outputContexts[elem]["parameters"]["edad"]

    conver_id = outputContexts[elem]["parameters"]["conver_id"]

    output = generate_response(entry, edad, conver_id, True)

    outputContexts[elem]["parameters"]["context"]["entry"]["ES"].append(output["entry"]["ES"])
    outputContexts[elem]["parameters"]["context"]["answer"]["ES"].append(output["answer"]["ES"])

    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["entry"]["EN"])
    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["answer"]["EN"])

    outputContexts[elem]["parameters"]["conver_id"] = output["conver_id"]

    answer = output["answer"]["ES"]

    response = {
        "fulfillmentText": answer,
        "output_contexts": []
    }

    save_conversation(
        outputContexts[elem]["parameters"]["context"],
        outputContexts[elem]["parameters"]["edad"], 
        outputContexts[elem]["parameters"]["date_ini"]
    )

    return response










app = FastAPI(version="1.0.0")

app.mount("/static", StaticFiles(directory=(BASE_PATH + "/static")))
templates = Jinja2Templates(directory=(BASE_PATH + "/templates"))


@app.get("/", response_class=HTMLResponse) 
def home(request: Request, dark_mode: str = ""):
    return templates.TemplateResponse(
        "home.html", 
        {
            "request": request,
            "dark_mode": dark_mode
        }
    )

@app.get("/chatbot", response_class=HTMLResponse)
def chatbot(request: Request, dark_mode: str = ""):
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

@app.get("/capture_image", response_class=HTMLResponse) 
def capture_image(request: Request, canal: str, dark_mode: str = ""):
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

@app.get("/interface_adult", response_class=HTMLResponse)
def interface_adult(request: Request, dark_mode: str = ""):
    return templates.TemplateResponse(
        "interface_adult.html", 
        {
            "request": request,
            "dark_mode": dark_mode
        }
    )

@app.get("/interface_child", response_class=HTMLResponse)
def interface_child(request: Request, dark_mode: str = ""):
    return templates.TemplateResponse(
        "interface_child.html", 
        {
            "request": request,
            "dark_mode": dark_mode
        }
    )

@app.get("/wakeup", response_class=PlainTextResponse)
def wakeup():
    return "Server ON"

@app.post("/setURL", response_class=PlainTextResponse)
def setURL(request: ServerURL):
    global SERVER_GPU_URL
    SERVER_GPU_URL = request.url
    return "URL fijada correctamente"

@app.post("/webhook_adult")
async def webhook_adult( request: Request):
    request_JSON = await request.json()

    print(request_JSON)

    intent = request_JSON["queryResult"]["intent"]["displayName"]

    if intent == "Talk":
        response = make_response_talk(request_JSON, TalkType.adult)
    elif intent == "Goodbye":
        # Implementar guardado del historial
        response = make_response_goodbye(request_JSON)

    print(response)

    return response

@app.post("/webhook_child")
async def webhook_child( request: Request):
    request_JSON = await request.json()

    print(request_JSON)

    intent = request_JSON["queryResult"]["intent"]["displayName"]

    if intent == "Talk":
        response = make_response_talk(request_JSON, TalkType.child)
    elif intent == "Goodbye":
        # Implementar guardado del historial
        response = make_response_goodbye(request_JSON)

    print(response)

    return response


def main():
    """! Entrada al programa principal."""

    uvicorn.run(app, host=HOST, port=PORT)



if __name__ == "__main__":
    main()
