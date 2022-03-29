#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import uvicorn
import os
from pathlib import Path
from pyngrok import ngrok
import nest_asyncio




app = FastAPI(version="1.0.0")

BASE_PATH = Path(__file__).resolve().parent

@app.get("/", response_class=PlainTextResponse)
def home():
    return "Server GPU ON"

@app.get("/Adulto", response_class=PlainTextResponse)
def adulto():
    return "Texto de prueba"



if __name__ == "__main__":

    with open(str(BASE_PATH/"config.json")) as file:
        config = json.load(file)

    port = eval(os.environ.get("PORT", config["port"]))

    public_url = ngrok.connect(port).public_url

    print(public_url)

    nest_asyncio.apply()

    uvicorn.run(app, port=port)
