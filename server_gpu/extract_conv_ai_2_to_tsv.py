#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset






if __name__ == "__main__":
    dataset = load_dataset("conv_ai_2")

    dialog = dataset["dialog"]

    print(dialog)


