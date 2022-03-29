#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from curses import meta
import json
from time import time
import requests
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from fastapi import FastAPI, Body, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel


BASE_PATH = Path(__file__).resolve().parent

with open(str(BASE_PATH/"config.json")) as file:
        config = json.load(file)

HOST = os.environ.get("HOST", config["host"])
PORT = eval(os.environ.get("PORT", config["port"]))
global SERVER_GPU_URL
SERVER_GPU_URL = os.environ.get("SERVER_GPU_URL", config["server_gpu_url"])







def make_response_welcome(request: Dict):
    POS_ID = 0
    POS_CONTEXT = 1
    POS_WELCOME_COMPLETE = 2

    outputContexts = request.get("queryResult").get("outputContexts")

    if SERVER_GPU_URL != "":          
        session_id = outputContexts[POS_ID]
        session_id["parameters"] = {
            "session_id": request.get("responseId")
        }
        outputContexts[POS_ID] = session_id

        entry = request.get("queryResult").get("queryText")

        query_json = {
            "entry": entry,
        }
        #answer = requests.post(str(SERVER_GPU_URL/"deduct"), json=query_json)
        answer = "Hola"

        context = outputContexts[POS_CONTEXT]
        context["parameters"] = {
            "context": f"[A]: {entry}\n[B]: {answer}"
        }
        outputContexts[POS_CONTEXT] = context

        welcome_complete = outputContexts[POS_WELCOME_COMPLETE]
        welcome_complete["parameters"] = {
            "welcome-complete": True
        }
        outputContexts[POS_WELCOME_COMPLETE] = welcome_complete
    else:
        outputContexts = []
        answer = "Servidor GPU no disponible"
    
    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

def make_response_deduct(request: Dict):
    POS_CONTEXT = 1
    POS_EDAD = 2
    POS_WELCOME_COMPLETE = 3
    POS_DEDUCT_COMPLETE = 4

    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    answer = entry

    if answer in ["Niño", "Adolescente", "Adulto"]:
        edad = outputContexts[POS_EDAD]
        edad["parameters"] = {
            "edad": answer
        }
        outputContexts[POS_EDAD] = edad

        deduct_complete = outputContexts[POS_DEDUCT_COMPLETE]
        deduct_complete["parameters"] = {
            "deduct-complete": True
        }
        outputContexts[POS_DEDUCT_COMPLETE] = deduct_complete

        outputContexts.pop(POS_WELCOME_COMPLETE)
    
    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

def make_response_talk(request: Dict):
    POS_CONTEXT = 1
    POS_EDAD = 2

    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    edad = outputContexts[POS_EDAD]["parameters"]["edad"]

    query_json = {
        "entry": entry,
    }
    answer = requests.post(str(SERVER_GPU_URL/edad), json=query_json)
    
    context = outputContexts[POS_CONTEXT]
    contextContent = context["parameters"]["context"]
    context["parameters"] = {
        "context": f"{contextContent}\n[A]: {entry}\n[B]: {answer}"
    }
    outputContexts[POS_CONTEXT] = context

    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

def make_response_goodbye(request: Dict):
    POS_CONTEXT = 1

    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    answer = "Adios"

    context = outputContexts[POS_CONTEXT]
    contextContent = context["parameters"]["context"]
    context["parameters"] = {
        "context": f"{contextContent}\n[A]: {entry}\n[B]: {answer}"
    }
    outputContexts[POS_CONTEXT] = context

    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response



app = FastAPI(version="1.0.0")


app.mount("/static", StaticFiles(directory=str(BASE_PATH/"static")))
templates = Jinja2Templates(directory=str(BASE_PATH/"templates"))


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
def chatbot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.get("/interface", response_class=HTMLResponse)
def interface(request: Request):
    return templates.TemplateResponse("interface.html", {"request": request})

@app.get("/wakeup", response_class=PlainTextResponse)
def wakeup():
    return "Server ON"

@app.get("/setURL")
def setURL(url: str):
    global SERVER_GPU_URL
    SERVER_GPU_URL = url
    return "URL fijada correctamente."

@app.post("/webhook")
async def webhook( request: Request):
    print("----------->")

    request_JSON = await request.json()

    intent = request_JSON.get("queryResult").get("intent").get("displayName")

    response = {}

    if intent == "Welcome":
        response = make_response_welcome(request_JSON)
    elif intent == "Deduct":
        response = make_response_deduct(request_JSON)
    elif intent == "Talk":
        response = make_response_talk(request_JSON)
    elif intent == "Goodbye":
        # Implementar guardado del historial
        response = make_response_goodbye(request_JSON)


    print(response)

    return response



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
