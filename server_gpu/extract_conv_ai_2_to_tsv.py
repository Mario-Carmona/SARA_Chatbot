#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import random
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_split", 
        type = float,
        help = ""
    )
    parser.add_argument(
        "train_file", 
        type = str,
        help = ""
    )
    parser.add_argument(
        "test_file", 
        type = str,
        help = ""
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    dataset = load_dataset("conv_ai_2")

    dialog = dataset["train"]["dialog"]

    conversaciones = []

    for conver in dialog:
        len_dialog = len(conver) if len(conver)%2 == 0 else len(conver)-1
        for j in range(0,len_dialog,2):
            entry = conver[j]["text"]
            response = conver[j+1]["text"]
            conversaciones.append(f"{entry}\t{response}")

    random.shuffle(conversaciones)

    div = int(len(conversaciones) * args.train_split)

    with open(args.train_file, 'w') as f:
        f.write('\n'.join(conversaciones[:div]))

    with open(args.test_file, 'w') as f:
        f.write('\n'.join(conversaciones[div:]))
