# Trabajo Fin de Grado

Repositorio que contiene mi Trabajo Fin de Grado realizado en la UGR durante el curso 21/22

## Índice

1. [Resumen](#resumen)
1. [Requisitos](#requisitos)
1. [Instalación servidor local](#instalación-servidor-local)

## Resumen

Este proyecto tiene como objetivo crear de forma automatizada un agente conversacional para “La noche europea de los investigadores” (ERN). Deberá poder contestar a las preguntas que le realicen los asistentes, adaptándose a sus edades y conocimientos de forma interactiva.

Para el desarrollo de este proyecto se ha construido un sistema con el que poder hacer uso del chatbot obtenido del entrenamiento. Además, se ha elaborado toda una estructura de programas para obtener información, entrenar a los modelos con esa información y finalmente desplegar estos modelos en el sistema mencionado anteriormente.

Para la adaptación del chatbot a distintos perfiles de usuario se ha elaborado un módulo que es capaz de deducir la edad de un usuario de forma explícita o de manera implícita a través de imágenes.

## Requisitos

Como el sistema se compone de dos servidores distintos, cada uno tendrá sus propios requisitos. Los requisitos de cada servidor se encuentran en archivos del tipo "requirements.txt", los cuales se utilizan para instalar todos los requisitos mediante pip.

Los requisitos del servidor de la APP se pueden ver con el siguiente enlace $\rightarrow$ [![Ver requisitos](https://img.shields.io/badge/Ver-Requisitos%20APP-inactive.svg)](https://github.com/Mario-Carmona/SARA_Chatbot/blob/main/app/requirements.txt)

- Usar como S.O. **Ubuntu** (Versión 20.04.3 ó superior) [<img src="./image_readme/logo_ubuntu.jpg" alt="Logo ubuntu" width="80" height="20"/>](https://ubuntu.com/download/desktop)
- Instalar **Visual Studio Code** (Versión 1.64.2 ó superior) [<img src="./image_readme/logo_vscode.jpg" alt="Logo vscode" width="20" height="20"/>](https://code.visualstudio.com/download)
- Instalar **npm** (Versión 6.14.4 ó superior) [<img src="./image_readme/logo_npm.png" alt="Logo npm" width="60" height="20"/>](https://www.npmjs.com/)
- Instalar **express** (Versión 4.17.3 ó superior) [<img src="./image_readme/logo_express.jpg" alt="Logo express" width="80" height="20"/>](https://www.npmjs.com/package/express)
- Instalar extensión **ngrok for VSCode** [<img src="./image_readme/logo_ngrok.png" alt="Logo ngrok" width="20" height="20"/>](https://marketplace.visualstudio.com/items?itemName=philnash.ngrok-for-vscode)


## Instalación servidor local

Para instalar el servidor local son necesarios los siguientes pasos:

1. 
