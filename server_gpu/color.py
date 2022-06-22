#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la definición de colores."""


##
# @file color.py
#
# @brief Programa para la definición de colores.
#
# @section description_main Descripción
# Programa para la definición de colores.
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.



class bcolors:
    """! Clase de colores para el terminal.
    Define los colores que pintan los caracteres escritos por salida estándar por pantalla.
    """

    OK = '\033[92m'         # GREEN
    WARNING = '\033[93m'    # YELLOW
    FAIL = '\033[91m'       # RED
    RESET = '\033[0m'       # RESET COLOR
