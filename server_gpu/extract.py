#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la extracción de información de la página Quora."""


##
# @file extract.py
#
# @brief Programa para la extracción de información de la página Quora.
#
# @section description_main Descripción
# Programa para la extracción de información de la página Quora.
#
# @section libraries_main Librerías/Módulos
# - Librería pandas (https://pandas.pydata.org/docs/)
#   - Acceso a la función DataFrame
# - Librería tqdm.auto (https://tqdm.github.io/)
#   - Acceso a la función tqdm
# - Librería estándar time (https://docs.python.org/3/library/time.html)
#   - Acceso a la función sleep
# - Librería selenium (https://www.selenium.dev/selenium/docs/api/py/api.html)
#   - Acceso a la función webdriver
#       - Acceso a la función Chrome
#       - Acceso a la función common.keys
#           - Acceso a la clase Keys
#       - Acceso a la función common.by
#           - Acceso a la clase By
# - Librería utils
#   - Acceso a la función save_csv
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import argparse
import pandas as pd
from utils import save_csv
from tqdm.auto import tqdm
import time

from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.by import By



def obtenerURLs(filename):
    """! Obtener URLs contenidas en el archivo
    
    @param filename  Archivo que contiene las URLs de los distintos Topics
    
    @return Lista de URLs de los distintos Topics
    """

    # Apertur en modo lectura del archivo
    with open(filename, mode='r', encoding='utf-8') as keywords_file:
        # Obtención de la lista de URLs
        keywords_list = keywords_file.readlines()

    return keywords_list


def open_URL(driver, url, time_sleep):
    """! Apertura de la página indicada.
    
    @param driver      Entorno web
    @param url         Dirección de la página que se quiere abrir
    @param time_sleep  Tiempo de espera tras la apertura de la página
    """

    # Apertur de la página
    driver.get(url)

    # Espera por la carga de la página
    time.sleep(time_sleep)


def login(args, driver):
    """! Descender por la página hasta carga todas los ejemplos de la página o hasta llegar al final de la misma.
    
    @param args    Argumentos del script
    @param driver  Entorno web
    """

    # Apertura de la página inicial de Quora
    open_URL(driver, "https://www.quora.com/", 1)

    # Obtención del campo del email
    email = driver.find_element(by=By.NAME, value="email")
    # Insertar información sobre el email
    email.send_keys(args.user) 
    
    # Obtención del campo de la contraseña
    passwd = driver.find_element(by=By.NAME, value="password")
    # Insertar información sobre la contraseña
    passwd.send_keys(args.pasw)

    # Espera por la inserción de información
    time.sleep(1)

    # Simular la pulsación de la tecla ENTER
    passwd.send_keys(Keys.ENTER)
    
    # Espera por la carga de la página principal de Quora
    time.sleep(5)


def scroll_down(driver, num_examples):
    """! Descender por la página hasta carga todas los ejemplos de la página o hasta llegar al final de la misma.
    
    @param driver        Entorno web
    @param num_examples  Número de ejemplos de la página
    """

    # Obtener la página
    page = driver.find_element(by=By.TAG_NAME, value="body")

    # Obtener los accesos a las preguntas de la página del Topic
    links = driver.find_elements(by=By.XPATH, value='.//a[@class = "q-box qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 KlcoI"]')

    # Contador del número de ejemplos cargados previo al descenso
    pre_num_examples = 0

    # Descenso inicial por la página
    page.send_keys(Keys.PAGE_DOWN)
        # Espera por el tiempo de carga
    time.sleep(0.2)
    page.send_keys(Keys.PAGE_DOWN)
        # Espera por el tiempo de carga
    time.sleep(0.2)

    # Obtener número de ejemplos actual
    num_examples_actual = len([link for link in links if link.text != "No answer yet"])

    # Descender hasta llegar al número máximo de ejemplos o al final de la página
    while pre_num_examples != len(links) and num_examples_actual < num_examples:
        # Actualización del número de ejemplos cargados previo al descenso
        pre_num_examples = len(links)

        # Descenso por la página
        page.send_keys(Keys.PAGE_DOWN)
            # Espera por el tiempo de carga
        time.sleep(0.2)
        
        # Obtener los accesos a las preguntas de la página del Topic
        links = driver.find_elements(by=By.XPATH, value='.//a[@class = "q-box qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 KlcoI"]')

        # Obtener número de ejemplos actual
        num_examples_actual = len([link for link in links if link.text != "No answer yet"])


def isAnswer(frase):
    """! Comprobar si el texto de una frase corresponde al texto de una respuesta.
    
    @param frase  Texto de una frase

    @return Confirmación o negación de que es una respuesta
    """

    # Comprobar si es una respuesta comprobando el color del texto
    return True if "rgba(40, 40, 41, 1)" == frase.value_of_css_property("color") else False


def obtenerTitle(box_question):
    """! Obtener el título que se encuentran en el contenido de la página.
    
    @param box_question  Lista con los elementos del contenido de la página

    @return Título de la página
    """

    # Obtención de la caja del título
    box_title = box_question.find_elements_by_xpath('//div[@class="q-box qu-borderAll qu-borderRadius--small qu-borderColor--raised qu-boxShadow--small qu-bg--raised"]')[0]
    
    # Obtener texto del título
    title = box_title.find_elements_by_xpath('//span[@class="q-box qu-userSelect--text"]')[0]

    return title


def obtenerAnswers(box_question):
    """! Obtener las respuesta que se encuentran en el contenido de la página.
    
    @param box_question  Lista con los elementos del contenido de la página

    @return Lista con las respuestas
    """

    # Obtención de las cajas de las respuestas
    box_answers = box_question.find_elements_by_xpath('//div[@class="q-box qu-pt--medium qu-hover--bg--darken"]')

    # Lista que contendrá las respuestas
    answers = []

    for i, box_answer in enumerate(box_answers):
        # Si la caja pertenece a una respuesta a la pregunta
        if not "Related" in box_answer.text:
            # Obtención del texto de la respuesta
            text = box_answer.find_elements_by_xpath('//div[@class="q-text qu-wordBreak--break-word"]')[i]

            # Si el texto corresponde a una respuesta
            if isAnswer(text):
                # Si el texto de la respuesta no se muestra en su totalidad
                if "(more)" in box_answer.text:
                    # Añadir un elemento nulo a la lista
                    answers.append(None)
                else:
                    # Añadir una nueva respuesta a la lista
                    answers.append(text)

    return answers

            
def obtenerContent(driver):
    """! Obtener el contenido de la página.
    
    @param driver       Entorno web

    @return Lista con el contenido de la página
    """

    # Obtención de las cajas con información
    box_question = driver.find_elements_by_xpath('//div[@class="q-box puppeteer_test_question_main"]')[0]
    box_question = box_question.find_elements_by_xpath('//div[@class="q-box"]')[0]

    # Obtención del título de la página
    title = obtenerTitle(box_question)

    # Obtención de las respuestas de la página
    answers = obtenerAnswers(box_question)

    # Creación de una lista conjunta con el título y la respuestas
    frases = [title] + answers

    return frases


def scroll_down_answers(driver, num_answers):
    """! Descender por la página hasta carga todas las respuestas de la página.
    
    @param driver       Entorno web
    @param num_answers  Número de respuestas de la página
    """

    # Obtener la página
    page = driver.find_element(by=By.TAG_NAME, value="body")

    # Variable que indica el fin del descenso
    finish = False

    while not finish:
        # Obtener el contenido de la página
        content = obtenerContent(driver)

        # Eliminar el titulo del contenido
        content = content[1:]
        
        # Si el número de respuestas actual es menor del total se desciende
        if len(content) < num_answers:
            page.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
        else:
            # En caso contrario, se indica el fin del descenso
            finish = True


def obtener_answers(driver):
    """! Obtener información de la página de la pregunta sibre un Topic.
    
    @param driver  Entorno web

    @return Pregunta de la página
    @return Lista de respuestas a la pregunta
    """

    # Obtener elemento de la página que indica el número de respuestas
    num_answers = driver.find_elements_by_xpath('//button[@class="q-click-wrapper ClickWrapper___StyledClickWrapperBox-zoqi4f-0 bIwtPb base___StyledClickWrapper-lx6eke-1 laIUvT   qu-active--bg--darken qu-active--textDecoration--none qu-borderRadius--pill qu-alignItems--center qu-justifyContent--center qu-whiteSpace--nowrap qu-userSelect--none qu-display--inline-flex qu-tapHighlight--white qu-textAlign--center qu-cursor--pointer qu-hover--bg--darken qu-hover--textDecoration--none"]')
    
    # Si hay alguna respuesta
    if len(num_answers) != 0:
        
        num_answers = [i for i in num_answers if "Answer" in i.text][0]
        num_answers = num_answers.text.split('\n')

        if len(num_answers) == 3:
            # Obtener el valor numérico del número de preguntas
            num_answers = int(num_answers[-1])

            # Si hay más de una respuesta
            if num_answers != 1:
                # Descenso por la página hasta cargar todas las respuestas
                scroll_down_answers(driver, num_answers)

            # Obtener el contenido de la página
            content = obtenerContent(driver)

            # Obtener el texto de cada elemento del contenido
            texts = [elem.text.replace("\n", " ") for elem in content if elem != None]
            
            # Devolución de la pregunta, y de la lista de respuestas
            return texts[0:1], texts[1:]
        else:
            return [], []
    else:
        # En caso contrario, se devuelven ambos elementos vacíos
        return [], []
        

def obtenerDatasetTopic(args, driver, topic):
    """! Generar el conjunto de datos de un Topic.
    
    @param args    Argumentos del script
    @param driver  Entorno web
    @param topic   URL del Topic

    @return Dataframe del Topic
    """

    # Extracción del nombre del Topic de su URL
    name_topic = topic.split('/')[-2].replace("-", " ")
    
    # Apertura de la URL del Topic en el entorno
    open_URL(driver, topic, 3)

    # Descenso por la página hasta llegar al máximo número de ejemplo o al final de la página
    scroll_down(driver, args.num_examples)

    # Listas que contendrán la información de cada una de las columnas
    list_topics = []
    list_subjects = []
    list_questions = []
    list_answers = []

    # Obtener los accesos a las preguntas de la página del Topic
    links = driver.find_elements(by=By.XPATH, value='.//a[@class = "q-box qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 KlcoI"]')

    # Obtener los enlaces de cada uno de los elementos de la lista
    enlaces = [link.get_attribute('href') for link in links if link.text != "No answer yet"][:args.num_examples]

    # Creación de la barra de progreso
    progress_bar = tqdm(range(len(enlaces)))

    for enlace in enlaces:
        # Apertura del enlace a la pregunta
        open_URL(driver, enlace, 3)

        # Obtención de la pregunta y las respuestas a la misma
        question, answers = obtener_answers(driver)

        # Actualización de la barra de progreso
        progress_bar.update(1)

        # Inserción de la información en las distintas listas
        list_topics += [name_topic] * len(answers)
        list_subjects += [name_topic] * len(answers)
        list_questions += question * len(answers)
        list_answers += answers

    # Creación del Dataframe que contendrá la información
    dataset = pd.DataFrame({
        "Topic": list_topics,
        "Subject": list_subjects,
        "Question": list_questions,
        "Answer": list_answers
    })

    return dataset
        

def obtenerDataset(args, keywords_list):
    """! Generar los datasets de los distintos Topics.
    
    @param args           Argumentos del script
    @param keywords_list  Lista de URLs de los Topics
    """

    # Obtener el entorno de Chrome
    driver = webdriver.Chrome() 
    
    # Realizar el login en la página de Quora
    login(args, driver)

    # Lista que contendrá los Dataframe de cada Topic
    group_datasets = []

    # Creación de la barra de progreso
    progress_bar = tqdm(range(len(keywords_list)))

    # Obtener el Dataframe de cada Topic
    for i, keyword in enumerate(keywords_list):
        # Obtención del Dataframe
        dataset = obtenerDatasetTopic(args, driver, keyword)

        # Creación del nombre del archivo que contendrá la información del Dataframe que se ha obtenido
        filename = '.'.join(args.output.split('.')[:-1])

        # Guardado de la información de Dataframe en el archivo
        save_csv(dataset, f"{filename}_{i}.csv")

        # Actualización de la barra de progreso
        progress_bar.update(1)

    # Cierre del entorno
    driver.close()


def main():
    """! Entrada al programa."""

    # Analizador de argumentos
    parser = argparse.ArgumentParser()

    # Añadir un argumento para el usuario de Quora
    parser.add_argument("-u", "--user", type=str, help="Usuario de la página Quora")

    # Añadir un argumento para la contraseña de Quora
    parser.add_argument("-p", "--pasw", type=str, help="Contraseña de la página Quora")

    # Añadir un argumento para el número máximo de ejemplos por cada Topic
    parser.add_argument("-n", "--num_examples", type=int, help="Número máximo de ejemplos por Topic")

    # Añadir un argumento para el archivo que contiene la lista de Topics
    parser.add_argument("-t", "--topics", help="El formato del archivo debe ser \'topics.txt\'")

    # Añadir un argumento para el archivo que contendrá el conjunto de datos resultante
    parser.add_argument("-f", "--output", help="El formato del archivo debe ser \'dataset.csv\'")

    # Obtención de los argumentos
    args = parser.parse_args()

    # Obtención de las URLs de los Topics
    keywords_list = obtenerURLs(args.topics)

    # Generar  los datasets de los distintos Topics
    obtenerDataset(args, keywords_list)


if __name__ == '__main__':
    main()    


    # ./prueba_extract.py -u "pepegarre008@gmail.com" -p "Macarse2000" -s 2 -t ./topics.txt -f ./salida.txt
