"""
import requests


url = "http://1ebc-34-121-126-47.ngrok.io/inferencia"
usuario = {
    "question": "A cat sat on a mat",
}
#respuesta = requests.post(url, json=usuario)

#print(respuesta.text)

requests.get("http://fd31-34-121-126-47.ngrok.io/shutdown")
"""


import requests

from flask import Flask, request

app = Flask(__name__) 

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

@app.route("/") 
def home(): 
  return "<h1>Bienvenido al servidor</h1>"
