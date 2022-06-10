import re
import urllib
import requests
from bs4 import BeautifulSoup
import pandas as pd

from tqdm.auto import tqdm


import time

import csv

from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils import save_csv

import argparse
 




def obtenerURLs(filename):
    with open(filename, mode='r', encoding='utf-8') as keywords_file:
        keywords_list = keywords_file.readlines()

    return keywords_list


def open_URL(driver, url, time_sleep):
    driver.get(url)
    time.sleep(time_sleep)


def login(args, driver):
    open_URL(driver, "https://www.quora.com/", 1)

    email = driver.find_element(by=By.NAME, value="email")
    email.send_keys(args.user) 
    
    passwd = driver.find_element(by=By.NAME, value="password")
    passwd.send_keys(args.pasw)

    time.sleep(1)

    passwd.send_keys(Keys.ENTER)
    
    time.sleep(5)


def scroll_down(driver, num_examples):
    page = driver.find_element(by=By.TAG_NAME, value="body")

    links = driver.find_elements(by=By.XPATH, value='.//a[@class = "q-box qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 KlcoI"]')

    pre_num_examples = 0

    page.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    page.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)


    num_answers = len([link for link in links if link.text != "No answer yet"])

    while pre_num_examples != len(links) and num_answers < num_examples:
        pre_num_examples = len(links)

        page.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        
        links = driver.find_elements(by=By.XPATH, value='.//a[@class = "q-box qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 KlcoI"]')

        num_answers = len([link for link in links if link.text != "No answer yet"])


def isAnswer(frase):
    return True if "rgba(40, 40, 41, 1)" == frase.value_of_css_property("color") else False


def obtenerTitle(box_question):
    box_title = box_question.find_elements_by_xpath('//div[@class="q-box qu-borderAll qu-borderRadius--small qu-borderColor--raised qu-boxShadow--small qu-bg--raised"]')[0]
    title = box_title.find_elements_by_xpath('//span[@class="q-box qu-userSelect--text"]')[0]

    return title


def obtenerAnswers(box_question):
    box_answers = box_question.find_elements_by_xpath('//div[@class="q-box qu-pt--medium qu-hover--bg--darken"]')

    answers = []

    for i, box_answer in enumerate(box_answers):
        if not "Related" in box_answer.text:
            text = box_answer.find_elements_by_xpath('//div[@class="q-text qu-wordBreak--break-word"]')[i]

            if isAnswer(text):
                if "(more)" in box_answer.text:
                    answers.append(None)
                else:
                    answers.append(text)

    return answers

            
def obtenerContent(driver, num_answers):
    box_question = driver.find_elements_by_xpath('//div[@class="q-box puppeteer_test_question_main"]')[0]
    box_question = box_question.find_elements_by_xpath('//div[@class="q-box"]')[0]

    title = obtenerTitle(box_question)

    answers = obtenerAnswers(box_question)

    frases = [title] + answers

    return frases


def scroll_down_answers(driver, num_answers):
    page = driver.find_element(by=By.TAG_NAME, value="body")

    finish = False

    while not finish:
        content = obtenerContent(driver, num_answers)

        # Eliminar titulo
        content = content[1:]
        
        if len(content) < num_answers:
            page.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
        else:
            finish = True


def obtener_answers(driver):
    num_answers = driver.find_elements_by_xpath('//button[@class="q-click-wrapper ClickWrapper___StyledClickWrapperBox-zoqi4f-0 bIwtPb base___StyledClickWrapper-lx6eke-1 laIUvT   qu-active--bg--darken qu-active--textDecoration--none qu-borderRadius--pill qu-alignItems--center qu-justifyContent--center qu-whiteSpace--nowrap qu-userSelect--none qu-display--inline-flex qu-tapHighlight--white qu-textAlign--center qu-cursor--pointer qu-hover--bg--darken qu-hover--textDecoration--none"]')
    if len(num_answers) != 0:
        num_answers = [i for i in num_answers if "Answer" in i.text][0]
        num_answers = num_answers.text.split('\n')

        if len(num_answers) == 3:
            num_answers = int(num_answers[-1])

            if num_answers != 1:
                scroll_down_answers(driver, num_answers)

            content = obtenerContent(driver, num_answers)

            texts = [elem.text.replace("\n", " ") for elem in content if elem != None]
            
            return texts[0:1], texts[1:]
        else:
            return [], []
    else:
            return [], []
        




def obtenerDatasetTopic(args, driver, topic):
    name_topic = topic.split('/')[-2].replace("-", " ")
    
    open_URL(driver, topic, 3)

    scroll_down(driver, args.num_examples)

    dataset = pd.DataFrame(columns = ["Topic","Subject","Question","Answer"])

    list_topics = []
    list_subjects = []
    list_questions = []
    list_answers = []

    links = driver.find_elements(by=By.XPATH, value='.//a[@class = "q-box qu-cursor--pointer qu-hover--textDecoration--underline Link___StyledBox-t2xg9c-0 KlcoI"]')

    enlaces = [link.get_attribute('href') for link in links if link.text != "No answer yet"][:args.num_examples]

    progress_bar = tqdm(range(len(enlaces)))

    for enlace in enlaces:
        open_URL(driver, enlace, 3)

        question, answers = obtener_answers(driver)

        progress_bar.update(1)

        list_topics += [name_topic] * len(answers)
        list_subjects += [name_topic] * len(answers)
        list_questions += question * len(answers)
        list_answers += answers

    dataset = pd.DataFrame({
        "Topic": list_topics,
        "Subject": list_subjects,
        "Question": list_questions,
        "Answer": list_answers
    })

    return dataset
        


def obtenerDataset(args, keywords_list):
    driver = webdriver.Chrome() 
    
    login(args, driver)

    group_datasets = []

    progress_bar = tqdm(range(len(keywords_list)))

    for i, keyword in enumerate(keywords_list):
        dataset = obtenerDatasetTopic(args, driver, keyword)

        filename = '.'.join(args.output.split('.')[:-1])
        save_csv(dataset, f"{filename}_{i}.csv")

        progress_bar.update(1)

    driver.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=str, help="")
    parser.add_argument("-p", "--pasw", type=str, help="")
    parser.add_argument("-n", "--num_examples", type=int, help="")
    parser.add_argument("-t", "--topics", help="")
    parser.add_argument("-f", "--output", help=" ")

    args = parser.parse_args()



    keywords_list = obtenerURLs(args.topics)

    obtenerDataset(args, keywords_list)
