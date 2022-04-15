#!/bin/bash

rm -r ./datasets/*
scp -r mcarmona@ngpu.ugr.es:/mnt/homeGPU/mcarmona/server_gpu/datasets/* ./datasets/
