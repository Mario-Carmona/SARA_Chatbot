#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from time import time
import os
from pathlib import Path
from typing import Dict
import requests

from pydantic import BaseModel

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn



BASE_PATH = str(Path(__file__).resolve().parent)

with open(BASE_PATH + "/config.json") as file:
    config = json.load(file)

HOST = os.environ.get("HOST", config["host"])
PORT = eval(os.environ.get("PORT", config["port"]))
global SERVER_GPU_URL
SERVER_GPU_URL = os.environ.get("SERVER_GPU_URL", config["server_gpu_url"])



class ServerURL(BaseModel):
    url: str



def make_response_welcome(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    if SERVER_GPU_URL != "":          
        entry = request.get("queryResult").get("queryText")

        query_json = {
            "entry": entry,
            "past_user_inputs": None,
            "generated_responses": None
        }
        """
        headers = {'content-type': 'application/json'}
        output = requests.post(SERVER_GPU_URL + "/deduct", json=query_json, headers=headers)
        """
        output = {
            "entry": {
                "ES": "Hola", 
                "EN": "Hello"
            },
            "answer": {
                "ES": "Hola", 
                "EN": "Hello"
            }
        }

        outputContexts[0]["parameters"] = {
            "context": {
                "entry": [output["entry"]["ES"]],
                "answer": [output["answer"]["ES"]]
            },
            "past_user_inputs": [output["entry"]["EN"]],
            "generated_responses": [output["answer"]["EN"]]
        }

        answer = output["answer"]["ES"]
    else:
        outputContexts = []
        answer = "Servidor GPU no disponible"
    
    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

def make_response_deduct_talk(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    if(not "edad" in outputContexts[0].get("parameters").keys()):
        return make_response_deduct(request)
    else:
        return make_response_talk(request)

def make_response_deduct(request: Dict):
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

def make_response_talk(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    edad = outputContexts[0]["parameters"]["edad"]

    past_user_inputs = outputContexts[0]["parameters"]["past_user_inputs"]
    generated_responses = outputContexts[0]["parameters"]["generated_responses"]

    query_json = {
        "entry": entry,
        "past_user_inputs": past_user_inputs,
        "generated_responses": generated_responses
    }
    headers = {'content-type': 'application/json'}
    output = requests.post(SERVER_GPU_URL + "/" + edad, json=query_json, headers=headers)

    print(output.content)

    time.sleep(2)

    outputContexts[0]["parameters"]["context"]["entry"].append(output["entry"]["ES"])
    outputContexts[0]["parameters"]["context"]["answer"].append(output["answer"]["ES"])
    outputContexts[0]["parameters"]["past_user_inputs"].append(output["entry"]["EN"])
    outputContexts[0]["parameters"]["generated_responses"].append(output["answer"]["EN"])

    answer = output["answer"]["ES"]

    response = {
        "fulfillmentText": answer,
        "output_contexts": outputContexts
    }

    return response

def make_response_goodbye(request: Dict):
    outputContexts = request.get("queryResult").get("outputContexts")

    entry = request.get("queryResult").get("queryText")

    edad = outputContexts[0]["parameters"]["edad"]

    past_user_inputs = outputContexts[0]["parameters"]["past_user_inputs"]
    generated_responses = outputContexts[0]["parameters"]["generated_responses"]

    query_json = {
        "entry": entry,
        "past_user_inputs": None,
        "generated_responses": None
    }
    """
    headers = {'content-type': 'application/json'}
    output = requests.post(SERVER_GPU_URL + "/" + edad, json=query_json, headers=headers)
    """
    output = {
        "entry": {
            "ES": "Adios", 
            "EN": "Bye bye"
        },
        "answer": {
            "ES": "Adios", 
            "EN": "Bye bye"
        }
    }

    outputContexts[0]["parameters"]["context"]["entry"].append(output["entry"]["ES"])
    outputContexts[0]["parameters"]["context"]["answer"].append(output["answer"]["ES"])
    outputContexts[0]["parameters"]["past_user_inputs"].append(output["entry"]["EN"])
    outputContexts[0]["parameters"]["generated_responses"].append(output["answer"]["EN"])

    answer = output["answer"]["ES"]

    response = {
        "fulfillmentText": answer,
        "output_contexts": []
    }

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
def home(request: Request):
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

@app.post("/setURL", response_class=PlainTextResponse)
def setURL(request: ServerURL):
    global SERVER_GPU_URL
    SERVER_GPU_URL = request.url
    return "URL fijada correctamente"

@app.post("/webhook")
async def webhook( request: Request):
    request_JSON = await request.json()

    print(request_JSON)

    intent = request_JSON["queryResult"]["intent"]["displayName"]

    if intent == "Welcome":
        response = make_response_welcome(request_JSON)
    elif intent == "Deduct And Talk":
        response = make_response_deduct_talk(request_JSON)
    elif intent == "Goodbye":
        # Implementar guardado del historial
        response = make_response_goodbye(request_JSON)

    print(response)

    return response



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
