#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset






if __name__ == "__main__":
    dataset = load_dataset("conv_ai_2")

    dialog = dataset["train"]["dialog"]

    conversaciones = []

    for conver in dialog:
        for j in range(0,len(conver),2):
            entry = conver[j]["text"]
            response = conver[j+1]["text"]
            conversaciones.append(f"{entry}\t{response}")

    print(conversaciones)


