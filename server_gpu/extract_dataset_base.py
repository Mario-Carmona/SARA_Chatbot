#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import random
from datasets import load_dataset




def extract_multi_woz_v22_dataset():
    dataset = load_dataset("multi_woz_v22", revision="master")

    conversaciones_train = []

    dialog = dataset["train"]["turns"]

    for conver in dialog:
        conver = conver["utterance"]

        len_dialog = len(conver) if len(conver)%2 == 0 else len(conver)-1
        for j in range(0,len_dialog,2):
            entry = conver[j]
            response = conver[j+1]
            conversaciones_train.append(f"{entry}\t{response}")


    conversaciones_valid = []

    dialog = dataset["validation"]["turns"]

    for conver in dialog:
        conver = conver["utterance"]

        len_dialog = len(conver) if len(conver)%2 == 0 else len(conver)-1
        for j in range(0,len_dialog,2):
            entry = conver[j]
            response = conver[j+1]
            conversaciones_valid.append(f"{entry}\t{response}")

    return conversaciones_train, conversaciones_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_file", 
        type = str,
        help = ""
    )
    parser.add_argument(
        "valid_file", 
        type = str,
        help = ""
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)



    multi_woz_v22_train, multi_woz_v22_valid = extract_multi_woz_v22_dataset()

    dataset_train = multi_woz_v22_train
    dataset_valid = multi_woz_v22_valid

    with open(args.train_file, 'w') as f:
        f.write('\n'.join(dataset_train))

    with open(args.valid_file, 'w') as f:
        f.write('\n'.join(dataset_valid))
