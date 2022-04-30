#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset






if __name__ == "__main__":
    dataset = load_dataset("conv_ai_2")

    dialog = dataset["train"]["dialog"]

    conversaciones = []

    for conver in dialog:
        len_dialog = len(conver) if len(conver)%2 == 0 else len(conver)-1
        for j in range(0,len_dialog,2):
            entry = conver[j]["text"]
            response = conver[j+1]["text"]
            conversaciones.append(f"{entry}\t{response}")

    print(conversaciones)


