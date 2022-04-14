#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
from datasets import load_dataset



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'list_sentiment', 
        type=list, 
        help=''
    )
    args = parser.parse_args()


    dataset = load_dataset("empathetic_dialogues")

    print(dataset["train"]['conv_id'])


