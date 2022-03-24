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
        return send_file('./templates/home.html')

    @app.route("/wakeup", methods=["GET"])
    def wakeup():
        return "Server ON"

    @app.route("/interface", methods=["GET", "POST"])
    def interface():
        return send_file('./templates/interface.html')

    @app.route("/chatbot", methods=["GET", "POST"])
    def chatbot():
        return send_file('./templates/chatbot.html')

    @app.route("/webhook", methods=["POST"])
    def webhook():
        req = request.get_json(silent=True, force=True)
        
        query_result = req.get("queryResult")
        intent = query_result.get("intent").get("displayName")

        if intent == "Welcome":
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

    with open("./config.json") as file:
        config = json.load(file)

    host = os.environ.get("HOST", config["host"])
    port = eval(os.environ.get("PORT", config["port"]))
    debug = eval(config["debug"])

    server = Server(host, port, debug)

    server.run()
