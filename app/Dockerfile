# Tomar como base una imagen con python 3.9 instalado
FROM python:3.9.10
# Copiar el contenido de la máquina local a la imagen
COPY . /app
# Establecer el directorio de trabajo en la imagen
WORKDIR /app
# Instalar los paquetes requeridos por el servidor
RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "main.py" ]