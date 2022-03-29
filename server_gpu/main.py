#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import uvicorn
import os
import requests
from pathlib import Path
from pyngrok import ngrok
import nest_asyncio
from pydantic import BaseModel




class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

class Entry(BaseModel):
    entry: str


app = FastAPI(version="1.0.0")

BASE_PATH = Path(__file__).resolve().parent

@app.get("/", response_class=PlainTextResponse)
async def home():
    return "Server GPU ON"

@app.post("/Adulto", response_class=PlainTextResponse)
async def adulto(request: Entry):

    print(request.entry)
    
    return "Texto de prueba"



if __name__ == "__main__":

    with open(str(BASE_PATH/"config.json")) as file:
        config = json.load(file)

    port = eval(os.environ.get("PORT", config["port"]))

    public_url = ngrok.connect(port).public_url

    print(bcolors.OK + "Public URL" + bcolors.RESET + ": " + public_url)

    print(bcolors.WARNING + "Enviando URL al controlador..." + bcolors.RESET)
    url = config["controller_url"]
    headers = {'content-type': 'application/json'}
    response = requests.post(url + "/setURL", json={"url": public_url}, headers=headers)
    print(bcolors.OK + "INFO" + bcolors.RESET + ": " + response)

    nest_asyncio.apply()

    uvicorn.run(app, port=port)
