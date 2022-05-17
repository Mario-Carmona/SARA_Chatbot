#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from time import time
import os
from pathlib import Path
from typing import Dict
import requests
import psycopg2
from datetime import datetime
import pytz
import enum

from pydantic import BaseModel

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn




class TalkType(enum.Enum):
    child = "child"
    adult = "adult"





BASE_PATH = str(Path(__file__).resolve().parent)

with open(BASE_PATH + "/config.json") as file:
    config = json.load(file)

HOST = os.environ.get("HOST", config["host"])
PORT = eval(os.environ.get("PORT", config["port"]))
global SERVER_GPU_URL
SERVER_GPU_URL = os.environ.get("SERVER_GPU_URL", config["server_gpu_url"])

SPAIN = pytz.timezone('Europe/Madrid')

class ServerURL(BaseModel):
    url: str

def obtenerElemContext(outputContexts):
    elem = [i for i, s in enumerate(outputContexts) if s["name"].__contains__("talk-followup")][0]

    if(not "parameters" in outputContexts[elem].keys()):
        outputContexts[elem]["parameters"] = {}

    return elem


def is_first_response(outputContexts):
    elem = obtenerElemContext(outputContexts)
    
    if(not "edad" in outputContexts[elem].get("parameters").keys()):
        return True
    else:
        return False



def generate_response(entry: str, edad: str, history=[]):
    query_json = {
        "entry": entry,
        "history": history
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

        outputContexts[elem]["parameters"]["history"] = output["history"]

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

    history = outputContexts[elem]["parameters"]["history"]

    output = generate_response(entry, edad, history)

    outputContexts[elem]["parameters"]["context"]["entry"]["ES"].append(output["entry"]["ES"])
    outputContexts[elem]["parameters"]["context"]["answer"]["ES"].append(output["answer"]["ES"])

    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["entry"]["EN"])
    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["answer"]["EN"])

    outputContexts[elem]["parameters"]["history"] = output["history"]

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

    history = outputContexts[elem]["parameters"]["history"]

    output = generate_response(entry, edad, history)

    outputContexts[elem]["parameters"]["context"]["entry"]["ES"].append(output["entry"]["ES"])
    outputContexts[elem]["parameters"]["context"]["answer"]["ES"].append(output["answer"]["ES"])

    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["entry"]["EN"])
    outputContexts[elem]["parameters"]["context"]["entry"]["EN"].append(output["answer"]["EN"])

    outputContexts[elem]["parameters"]["history"] = output["history"]

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

@app.middleware("http")
def add_process_time_header(request: Request, call_next):
    start_time = time()
    response = call_next(request)
    process_time = time() - start_time
    print(f"Tiempo del proceso: {process_time}")
    return response

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



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
