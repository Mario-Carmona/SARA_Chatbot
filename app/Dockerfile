# Tomar como base una imagen con python 3.9 instalado
FROM python:3.9.10-alpine3.15
# Establecer el directorio de trabajo en la imagen
WORKDIR /app
# Copiar el contenido de la máquina local a la imagen
COPY ./* /app/
# Instalar los paquetes requeridos por el servidor
RUN pip install -r requirements.txt
# Ejecutar el script que inicia el servidor
CMD python main.py