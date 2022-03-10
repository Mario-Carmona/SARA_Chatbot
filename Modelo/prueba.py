import requests

url = "http://f0bc-34-86-152-52.ngrok.io/inferencia"
usuario = {
    "question": "A cat sat on a mat",
}
respuesta = requests.post(url, json=usuario)

print(respuesta.text)
