#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from flask import Flask, request, send_file


app = Flask(__name__)

@app.route("/") 
def home(): 
  return "<h1>Bienvenido al servidor</h1>"

@app.route("/interface", methods=["GET", "POST"])
def interface():
    return send_file('./templates/interface.html')

@app.route("/webhook", methods=["POST"])
def webhook():
  req = request.get_json(silent=True, force=True)
  
  query_result = req.get("queryResult")
  intent = query_result.get("intent").get("displayName")

  if intent == "Prueba":
    url = "http://4416-35-184-60-16.ngrok.io/inferencia"
    question = query_result.get("queryText")
    usuario = {
        "question": question,
    }
    fulfillmentText = requests.post(url, json=usuario)
  else:
    fulfillmentText = "Sin respuesta"

  return {
    "fulfillmentText": fulfillmentText,
    "source": "webhookdata"
  }

if __name__ == "__main__":
    app.run()
