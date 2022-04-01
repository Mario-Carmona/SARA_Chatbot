#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import sys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "txtfile", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.txt\'"
    )
    parser.add_argument(
        "csvfile", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )

    try:
        args = parser.parse_args()
        assert args.txtfile.split('.')[-1] == "txt"
        assert args.csvfile.split('.')[-1] == "csv"
    except:
        parser.print_help()
        sys.exit(0)

    with open(args.txtfile, "r") as txtfile:
        lines = txtfile.read().split('\n')
    
    for i in range(len(lines)):
        lines[i] = lines[i].split('|')

    csvfile = pd.DataFrame(lines[1:], columns=lines[0])

    csvfile.to_csv(args.csvfile)
