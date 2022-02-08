#!/bin/bash

url_zip="https://es.overleaf.com/project/620014156fc9a5493bab6c2c/download/zip"
tiempo_espera=3
carpe_imagenes="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/imagenes"
carpe_portada="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/portada"
carpe_prefacios="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/prefacios"
archi_tex="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/proyecto.tex"
archi_pdf="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/proyecto.pdf"
archi_zip_1="/home/macarse/Descargas/Proyecto_TFG.zip"
archi_zip_2="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/Proyecto_TFG.zip"
carpe_memoria="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria"
archi_aux="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/proyecto.aux"
archi_log="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/proyecto.log"
archi_out="/home/macarse/Escritorio/Universidad/2ºCuatrimestre/TFG/Proyecto/Memoria/proyecto.out"


$(xdg-open $url_zip)

sleep $tiempo_espera

$(rm -rf $carpe_imagenes $carpe_portada $carpe_prefacios $archi_tex $archi_pdf)

$(mv $archi_zip_1 $carpe_memoria)

$(unzip $archi_zip_2 -d $carpe_memoria)

$(rm $archi_zip_2)

$(pdflatex $archi_tex --output-directory=$carpe_memoria)

$(rm $archi_aux $archi_log $archi_out)


if [[ $# -eq 1 ]]; then
	$(git add *)
	$(git commit -m "$1")
	$(git push)
fi

