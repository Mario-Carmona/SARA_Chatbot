
"""! @brief Script para la definición de funciones auxiliares."""


##
# @file utils.py
#
# @brief Programa para la definición de funciones auxiliares.
#
# @section description_main Descripción
# Programa para la definición de funciones auxiliares.
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.



def save_csv(dataset, filepath):
    """! Ordenar un dataset.
    
    @param dataset   Dataframe que contiene los datos
    @param filepath  Ruta al archivo donde guardar el Dataframe
    """

    # Variable que contendrá la líneas a escribir en el archivo
    text = []

    # Generación de la línea que define las columnas del Dataframe
    text.append(','.join(list(dataset.columns.values)))

    # Obtención de las líneas de los datos del Dataframe
    for i in range(dataset.shape[0]):
        try:
            line = []
            for elem in dataset.iloc[i]:
                new_elem = elem.replace("\"", "'")
                line.append(f"\"{new_elem}\"")
            text.append(','.join(line))
        except:
            pass

    # Escritura de las líneas en el archivo
    with open(filepath, 'w') as f:
        f.write('\n'.join(text))
