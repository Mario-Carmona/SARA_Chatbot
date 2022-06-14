/* Obteción del elemento que captura la imagen */
var video = document.getElementById('video');

/* Obtener acceso a la cámara del dispositivo */
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    /* Petición de uso del dispositivo de video */
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        /* Asigna como fuente de video al stream que se genera con la sucesiva toma de fotos con el dispositivo */
        video.srcObject = stream;
        /* Reproducción del stream asignado como fuente */
        video.play();
    });
}


/*****************************************/

/* El código de esta sección sólo está enfocada a los dispositivos móviles, dado que la pantalla de los teléfonos móviles es pequeña la imagen no se puede gestionar de la misma forma en que se hace con el resto de dispositivos. En concreto, para los teléfonos móviles se accede directamente a su cámara, lo cuál abre la aplicación del sistema de la cámara, y dentro de esta aplicación se acepta o se rechaza la imagen tomada */

/* Cámara del dispositivo móvil */
const camera = document.querySelector('#camera');

/* Definir evento de cambio en la imagen tomada con la cámara */
camera.addEventListener('change', function(e) {
    /* Se muestra una ventana emergente que pregunta si se quiere enviar la foto tomada */
    Swal.fire({
        title: '¿Deseas enviar esta foto?',
        text: "¡No podrás revertir tu decisión!",
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#00FF24',
        cancelButtonColor: '#d33',
        confirmButtonText: 'Enviar',
        cancelButtonText: 'No enviar'
    }).then((result) => {
        /* Si se da permiso para enviar la foto */
        if (result.isConfirmed) {
            /* Obtención de un lector de archivos */
            var reader = new FileReader();

            /* Lectura de la imagen tomada en formato base64 */
            reader.readAsDataURL(e.target.files[0]);

            /* Función que se ejecuta tras la lectura del archivo */
            reader.onload = function() {
                /* Obtención de la URL del servidor GPU */
                var url = document.getElementById('url').innerText;

                /* Elaboración de la URL completa que se va a abrir */
                url = url + '/' + document.getElementById('deduct').innerText;

                /* Obtención de la imagen leida en formato base64 */
                var imgBase64 = reader.result;

                /* Inicialización de una petición HTTPS */
                const Http = new XMLHttpRequest();

                /* Apertura de una petición POST con dirección a la sección de deducción del servidor GPU */
                Http.open("POST", url, true);

                /* Fijar la cabecera de la petición */
                Http.setRequestHeader("Content-Type", "application/json");

                /* Función para gestionar la respuesta a la petición */
                Http.onreadystatechange = function() {
                    /* Obtención de la edad deducida */
                    var age = Http.responseText

                    /* Si el canal elegido es la Web */
                    if (document.getElementById('canal').innerText == "web") {
                        /* Se abre la URL hacia la interfaz web del chatbot */
                        openURL('./' + document.getElementById('web_' + age).innerText);
                    } else if (document.getElementById('canal').innerText == "telegram") {
                        /* Si el canal elegido es Telegram */

                        /* Se abre la URL hacia el chatbot de Telegram */
                        openURL(document.getElementById('telegram_' + age).innerText);
                    }
                };

                /* Elaboración de los datos que se enviarán en la petición */
                var data = {
                    imagen: imgBase64
                }

                /* Envío de la petición POST */
                Http.send(JSON.stringify(data));

                /* Obtener el elemento que muestra la carga de la página */
                var loading = document.getElementById('loading');

                /* Activación del movimiento del elemento de carga */
                loading.classList.toggle('active');
            };
        } else {
            /* En caso contrario */

            /* Se descarta la foto tomada */
            e.target.value = "";
        }
    });
});

/*****************************************/


/* Botón para la captura de la imagen */
var snap = document.getElementById("snap");

/* Definir evento de click para el botón de captura */
snap.onclick = function() {
    /* Captura de la imagen */
    snapPhoto();
};

/* Botón para el envío de la imagen */
var send = document.getElementById("send");

/* Definir evento de click para el botón de envío */
send.onclick = function() {
    /* Envío de la imagen */
    sendPhoto();
};

/* Link de la barra de navegación para ir a la página principal */
var link_inicio = document.getElementById("link_inicio");

/* Definir evento de click para el link para ir a la página principal */
link_inicio.onclick = function() {
    /* Ir a la página principal */
    openURL('./');
};

/* Link de la barra de navegación lateral para ir a la página principal */
var link_inicio_lateral = document.getElementById("link_inicio_lateral");

/* Definir evento de click para el link lateral para ir a la página principal */
link_inicio_lateral.onclick = function() {
    /* Ir a la página principal */
    openURL('./');
};