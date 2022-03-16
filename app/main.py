#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
import os
from flask import Flask, request, send_file


class Server:
    app = Flask(__name__)

    def __init__(self, host, port, debug):
        self.host = host
        self.port = port
        self.debug = debug

    def run(self):
        Server.app.run(debug=self.debug, host=self.host, port=self.port)

    @app.route("/") 
    def home(): 
        return "<h1>Bienvenido al servidor</h1>"

    @app.route("/wakeup", methods=["GET"])
    def wakeup():
        return "Server ON"

    @app.route("/set_url_server_gpu", methods=["POST"])
    def set_url_server_gpu():
        return "Server ON"

    @app.route("/interface", methods=["GET", "POST"])
    def interface():
        return send_file('./templates/interface.html')

    @app.route("/webhook", methods=["POST"])
    def webhook():
        req = request.get_json(silent=True, force=True)
        
        query_result = req.get("queryResult")
        intent = query_result.get("intent").get("displayName")

        if intent == "Prueba":
            url = config["inference_url"]
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

    with open("config.json") as file:
        config = json.load(file)

    host = config["host"]
    port = int(config["port"])
    debug = bool(config["debug"])

    server = Server(host, port, debug)

    server.run()
