#!/bin/bash

echo -e "\e[32m→ Realizando instalación de 'npm' \e[0m"
sudo apt install npm
echo -e "\e[32m→ Instalación de 'npm' finalizada \e[0m"

echo -e "\e[32m→ Realizando instalación de 'express' mediante npm \e[0m"
npm i express
echo -e "\e[32m→ Instalación de 'express' finalizada \e[0m"

echo -e "\e[32m→ Realizando instalación de 'ngrok for VSCode' mediante VSCode \e[0m"
code --install-extension philnash.ngrok-for-vscode
echo -e "\e[32m→ Instalación de 'ngrok for VSCode' finalizada \e[0m"

