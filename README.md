# Trabajo Fin de Grado

Repositorio que contiene mi Trabajo Fin de Grado realizado en la UGR durante el curso 21/22

## Índice

1. [Resumen](#resumen)
1. [Requisitos](#requisitos)
1. [Instalación](#instalación)
	1. [Instalación del servidor de la Vista y el Controlador](#instalación-del-servidor-de-la-vista-y-el-controlador)
	1. [Instalación del servidor del Modelo](#Instalación-del-servidor-del-Modelo)
1. [Inicio del sistema](#Inicio-del-sistema)
1. [Generación de los conjuntos de datos de entrenamiento](#Generación-de-los-conjuntos-de-datos-de-entrenamiento)
1. [Entrenamiento de los modelos](#Entrenamiento-de-los-modelos)
1. [Contacto](#Contacto)

## Resumen

Este proyecto tiene como objetivo crear de forma automatizada un agente conversacional para “La noche europea de los investigadores” (ERN). Deberá poder contestar a las preguntas que le realicen los asistentes, adaptándose a sus edades y conocimientos de forma interactiva.

Para el desarrollo de este proyecto se ha construido un sistema con el que poder hacer uso del chatbot obtenido del entrenamiento. Además, se ha elaborado toda una estructura de programas para obtener información, entrenar a los modelos con esa información y finalmente desplegar estos modelos en el sistema mencionado anteriormente.

Para la adaptación del chatbot a distintos perfiles de usuario se ha elaborado un módulo que es capaz de deducir la edad de un usuario de forma explícita o de manera implícita a través de imágenes.

## Requisitos

Como el sistema se compone de dos servidores distintos, cada uno tendrá sus propios requisitos. Los requisitos de cada servidor se encuentran en archivos del tipo "requirements.txt", los cuales se utilizan para instalar todos los requisitos mediante pip.

Los requisitos del servidor de la APP se pueden ver con el siguiente enlace [![Ver requisitos](https://img.shields.io/badge/Ver-Requisitos%20APP-inactive.svg)](https://github.com/Mario-Carmona/SARA_Chatbot/blob/main/app/requirements.txt).

Los requisitos del servidor GPU se pueden ver con el siguiente enlace [![Ver requisitos](https://img.shields.io/badge/Ver-Requisitos%20Server%20GPU-inactive.svg)](https://github.com/Mario-Carmona/SARA_Chatbot/blob/main/server_gpu/requirements.txt).

Como sistema operativo se ha utilizado en todo momento Ubuntu (![Ubuntu](https://img.shields.io/badge/Ubuntu-v20.04.3-orange.svg)).


## Instalación

Para ambas instalaciones se deberá crear un repositorio propio que contenga el código de nuestro repositorio en Github

### Instalación del servidor de la Vista y el Controlador

Una vez tenemos el repositorio debemos clonar el repositorio en nuestra máquina. Suponiendo que hemos creado una app en nuestra cuenta de Heroku, deberemos actualizar la información de nuestro Workflow para el servidor de la Vista y el Controlador con la información de nuestra cuenta y nuestra app; como son el email de nuestra cuenta de Heroku, el nombre de la app y la key de la API de nuestra cuenta.

Una vez tenemos actualizada la información del Workflow, al realizar un commit sobre el código de la carpeta app de nuestro nuevo repositorio, se realizará el despliegue de la app en la plataforma Heroku.

### Instalación del servidor del Modelo

Esta instalación se realizará en una máquina distinta con GPU, suponiendo que la máquina utilizada para la primera instalación no dispone de GPU. La instalación consistirá en clonar el repositorio en esta máquina con GPU. Adicionalmente se deben descargar los modelos utilizados para todas las funcionalidades del Modelo. Los modelos usados son los siguientes:

- [facebook/blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill/tree/main)
- [mrm8488/t5-base-finetuned-question-generation-ap](https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap/tree/main)
- [google/pegasus-xsum](https://huggingface.co/google/pegasus-xsum/tree/main)
- [nateraw/vit-age-classifier](https://huggingface.co/nateraw/vit-age-classifier/tree/main)

Estos modelos deberán ser descargados y movidos a la misma altura que la raíz del repositorio de trabajo.

Finalmente se deberá ejecutar el script de shell llamado _setup.sh_. Los cinco primeros comandos de este script deben ser eliminados si no se utilizan los servidor GPU del Instituto DaSCI.

## Inicio del sistema

Previamente, para poder iniciar el sistema se deberán descargar los modelos entrenados, lo cuales se encuentra en una [carpeta en Google Drive](https://drive.google.com/drive/folders/1jZz3FZ-VNf4JnqToJ7cFQXa-gLx9I1Fs?usp=sharing). Al descargarlos se deberá actualizar la configuración del servidor con las nuevas rutas de los modelos entrenados.

Además, si se quiere ver como están hechos por dentro los chatbots de Dialogflow, contactar a través del correo mcs2000carmona@correo.ugr.es.

El servidor que se ha desplegado en Heroku se estará ejecutando indefinidamente. Y en cuanto al servidor del Modelo para iniciar su ejecución se deberá ejecutar el siguiente comando a la altura de la carpeta _server\_gpu_:

~~~
    deepspeed --num_gpus 1 main.py --config_file configs/config_server.json
~~~

Tras la ejecución del comando y la finalización de la carga de todos los modelos que utiliza este servidor, el sistema estará disponible en su plenitud.

## Generación de los conjuntos de datos de entrenamiento

En primer lugar, se deberá generar un conjunto de datos inicial mediante la extracción de información de la página Quora. La extracción del conjunto de datos inicial se realiza mediante el siguiente comando:

~~~
    ./prueba_extract.py -u "<Correo de Quora>" -p "<Contraseña de Quora>" -n <Número de ejemplos por Topic> -t <Ruta al archivo con la lista de Topics> -f <Ruta al archivo de salida>
~~~

Una vez finalizada la extracción del conjunto de datos inicial, procedemos a generar los conjuntos de datos de entrenamiento. Además del conjunto de datos extraído de Quora se pueden generar conjuntos de datos creados a mano, los cuales se concatenaran al conjunto de datos de Quora durante el pre procesado. Deberemos actualizar el archivo de configuración de generación de conjuntos de datos, _config\_genDataset.json_, y posteriormente ejecutar el siguiente comando:

~~~
    python generate_dataset.py configs/config_genDataset.json
~~~

Tras las ejecución de este comando, obtendremos los conjuntos de datos de entrenamiento en la ruta que se ha indicado en el archivo de configuración.


## Entrenamiento de los modelos

Suponiendo que se han generado los conjuntos de datos de entrenamiento. Primeramente se deberá actualizar el archivo de configuración para el entrenamiento de cierto rango de edad, y posteriormente se ejecuta el siguiente comando:

~~~
    python fine_tuning.py configs/config_finetuning_child.json
~~~

En el caso del anterior comando, tras su finalización se obtendrá un modelo conversacional entrenado para el rango de edad de niños.


## Contacto

El sitio web que se aloja en Heroku siempre se encuentra activo, pero el servidor GPU no se encuentra siempre activo dado que no son recursos privados, sino que se deben compartir con el resto de estudiantes. En caso de querer hacer uso del sistema para realizar pruebas con él o para poder evaluarlo, solamente debe contactar con el administrador del sistema a través del correo electrónico mcs2000carmona@correo.ugr.es. Al enviar el correo, el administrador notificará el periodo de tiempo durante el cuál el sistema estará activado.


