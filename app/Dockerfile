# Tomar como base una imagen con python 3.9 instalado
FROM python:3.9
# Establecer el directorio de trabajo en la imagen
WORKDIR /home/app
# Copiar el contenido de la máquina local a la imagen
COPY ./* /home/app/
# Instalar los paquetes requeridos por el servidor
RUN pip install -r requirements.txt
# Ejecutar el script que inicia el servidor
CMD python __init__.py