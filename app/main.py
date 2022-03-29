#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from time import time
import requests
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn


BASE_PATH = Path(__file__).resolve().parent

with open(str(BASE_PATH/"config.json")) as file:
        config = json.load(file)

HOST = os.environ.get("HOST", config["host"])
PORT = eval(os.environ.get("PORT", config["port"]))
SERVER_GPU_URL = os.environ.get("SERVER_GPU_URL", config["server_gpu_url"])


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
    SERVER_GPU_URL = url
    return "URL fijada correctamente."

@app.post("/webhook")
async def webhook(request: Request):
    req = await request.json()
    
    query_result = req.get("queryResult")
    intent = query_result.get("intent").get("displayName")

    outputContexts = query_result.get("outputContexts")
    outputContexts2 = query_result.get("outputContexts")

    if intent == "Welcome":
        POS_ID = 1
        POS_CONTEXT = 2

        if SERVER_GPU_URL != "":
            session_id = outputContexts[POS_ID]
            session_id["parameters"] = {
                "session_id": req.get("responseId")
            }
            outputContexts[POS_ID] = session_id

            entry = query_result.get("queryText")

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
        else:
            outputContexts = []
            answer = "Servidor GPU no disponible"

    elif intent == "Deduct":
        POS_ID = 1
        POS_CONTEXT = 2
        POS_EDAD = 3

        entry = query_result.get("queryText")

        answer = entry

        edad = outputContexts[POS_EDAD]
        edad["parameters"] = {
            "edad": entry
        }
        outputContexts[POS_EDAD] = edad

    elif intent == "Talk":
        POS_ID = 1
        POS_CONTEXT = 2
        POS_EDAD = 3

        entry = query_result.get("queryText")
        edad = outputContexts[POS_EDAD]["parameters"]["edad"]

        query_json = {
            "entry": entry,
        }
        answer = requests.post(str(SERVER_GPU_URL/edad), json=query_json)
        
        context = outputContexts[POS_CONTEXT]
        context["parameters"] = {
            "context": f"{context}\n[A]: {entry}\n[B]: {answer}"
        }
        outputContexts[POS_CONTEXT] = context

    elif intent == "Goodbye":
        # Implementar guardado del historial
        pass


    webhookResponse = {
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [
                        answer
                    ]
                }
            }
        ],
        "outputContexts": outputContexts2
    }

    return webhookResponse



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
