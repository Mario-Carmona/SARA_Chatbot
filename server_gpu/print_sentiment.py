#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset



if __name__ == "__main__":

    dataset = load_dataset("empathetic_dialogues")

    sentiments = dataset["train"]["context"]

    list_set = set(sentiments)
    sentiments = (list(list_set))

    print(sentiments)
