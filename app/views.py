#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __init__ import app
import requests
from flask import request, render_template


@app.route("/") 
def home(): 
  return "<h1>Bienvenido al servidor</h1>"

@app.route("/interface", methods=["GET", "POST"])
def interface():
    return render_template('template/interface.html')

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