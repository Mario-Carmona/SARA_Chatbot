#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from time import time
from importlib_metadata import version
import requests
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
#from flask import Flask, request, send_file
from pathlib import Path



app = FastAPI(version="1.0.0")

BASE_PATH = Path(__file__).resolve().parent
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

@app.post("/webhook")
async def webhook(request: Request):
    req = await request.json()
    
    query_result = req.get("queryResult")
    intent = query_result.get("intent").get("displayName")

    if intent == "Welcome":
        url = config["inference_url"]
        answer = requests.post(url)
        print(answer)
        
        """
        outputContexts = query_result.get("outputContexts")
        name = outputContexts[0].get("name")
        session_id = req.get("responseId")
        
        webhookResponse = {
            "outputContexts": [
                {
                    "name": name,
                    "lifespanCount": 5,
                    "parameters": {
                        "session_id": session_id
                    }
                }
            ]
        }
        """
    elif intent == "Talk":
        url = config["inference_url"]
        question = query_result.get("queryText")
        query_json = {
            "question": question,
        }
        answer = requests.post(url, json=query_json)

        outputContexts = query_result.get("outputContexts")

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
            "outputContexts": outputContexts
        }
    elif intent == "Goodbye":
        # Implementar guardado del historial
        pass
    else:
        webhookResponse = {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            "Sin respuesta"
                        ]
                    }
                }
            ]
        }

    return webhookResponse



if __name__ == "__main__":

    with open(str(BASE_PATH/"config.json")) as file:
        config = json.load(file)

    host = os.environ.get("HOST", config["host"])
    port = eval(os.environ.get("PORT", config["port"]))

    uvicorn.run(app, host=host, port=port)
