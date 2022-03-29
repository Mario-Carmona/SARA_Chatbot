#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from time import time
import os
from pathlib import Path
from typing import Dict
import requests

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel


BASE_PATH = str(Path(__file__).resolve().parent)

with open(BASE_PATH + "/config.json") as file:
        config = json.load(file)

HOST = os.environ.get("HOST", config["host"])
PORT = eval(os.environ.get("PORT", config["port"]))
global SERVER_GPU_URL
SERVER_GPU_URL = os.environ.get("SERVER_GPU_URL", config["server_gpu_url"])



class ServerURL(BaseModel):
    url: str



async def make_response_welcome(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    if SERVER_GPU_URL != "":          
        entry = request.get("queryResult").get("queryText")

        query_json = {
            "entry": entry,
        }
        """
        headers = {'content-type': 'application/json'}
        answer = requests.post(SERVER_GPU_URL + "/deduct", json=query_json, headers=headers).content.decode('utf-8')
        """
        answer = "Hola"

        outputContexts[0]["parameters"] = {
            "context": f"[A]: {entry}\n[B]: {answer}"
        }
    else:
        outputContexts = []
        answer = "Servidor GPU no disponible"
    
    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

async def make_response_deduct_talk(request: Dict):
    POS_EDAD = 2

    outputContexts = request.get("queryResult").get("outputContexts")

    if(not "edad" in outputContexts[0].get("parameters").keys()):
        return await make_response_deduct(request)
    else:
        return await make_response_talk(request)

async def make_response_deduct(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    answer = entry

    if answer in ["Niño", "Adolescente", "Adulto"]:
        outputContexts[0]["parameters"]["edad"] = answer
    
    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

async def make_response_talk(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    edad = outputContexts[0]["parameters"]["edad"]

    query_json = {
        "entry": entry,
    }
    headers = {'content-type': 'application/json'}
    answer = requests.post(SERVER_GPU_URL + "/" + edad, json=query_json, headers=headers)

    print(answer)
    print(answer.content)
    print(answer.content.decode('utf-8'))

    context = outputContexts[0]["parameters"]["context"]

    outputContexts[0]["parameters"]["context"] = f"{context}\n[A]: {entry}\n[B]: {answer}"

    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

async def make_response_goodbye(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    answer = "Adios"

    context = outputContexts[0]["parameters"]["context"]

    outputContexts[0]["parameters"]["context"] = f"{context}\n[A]: {entry}\n[B]: {answer}"

    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response



app = FastAPI(version="1.0.0")


app.mount("/static", StaticFiles(directory=(BASE_PATH + "/static")))
templates = Jinja2Templates(directory=(BASE_PATH + "/templates"))


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    process_time = time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", response_class=HTMLResponse) 
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.get("/interface", response_class=HTMLResponse)
async def interface(request: Request):
    return templates.TemplateResponse("interface.html", {"request": request})

@app.get("/wakeup", response_class=PlainTextResponse)
async def wakeup():
    return "Server ON"

@app.post("/setURL", response_class=PlainTextResponse)
async def setURL(request: ServerURL):
    global SERVER_GPU_URL
    SERVER_GPU_URL = request.url
    return "URL fijada correctamente"

@app.post("/webhook")
async def webhook( request: Request):
    print("----------->")

    request_JSON = await request.json()

    intent = request_JSON.get("queryResult").get("intent").get("displayName")

    if intent == "Welcome":
        response = await make_response_welcome(request_JSON)
    elif intent == "Deduct And Talk":
        response = await make_response_deduct_talk(request_JSON)
    elif intent == "Goodbye":
        # Implementar guardado del historial
        response = await make_response_goodbye(request_JSON)

    print(response)

    return response



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
