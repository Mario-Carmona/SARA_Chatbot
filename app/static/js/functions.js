/* Función para la obtención del estado del switch 

    @return Estado del switch
*/
function obtain_status_dark_mode() {
    var buttonSwitch = document.querySelector('#switch_dark_mode');
    return buttonSwitch.classList.value;
}


/* Función para abrir una URL en el navegador 

    @param url    URL que se quiere abrir
    @param canal  Canal que se está usando para usar el chatbot (Web o Telegram)
*/
function openURL(url, canal = '') {
    /* Variables que contendrá la URL completa */
    var url_completa = '';

    /* Dependiendo de si se indica el canal o no, la URL se formará de forma distinta */
    if (canal == '') {
        /* Si no se indica el canal la URL completa está compuesta de la URL y el parámetro que indica el estado del switch del modo oscuro */
        url_completa = url + '?dark_mode=' + obtain_status_dark_mode();
    } else {
        /* Si no se indica el canal la URL completa está compuesta de la URL, el parámetro que indica el canal que se está usando, y el parámetro que indica el estado del switch del modo oscuro */
        url_completa = url + '?canal=' + canal + '&dark_mode=' + obtain_status_dark_mode();
    }

    /* Apertura de la URL completa */
    window.open(url_completa, "_self");
}


/* Función para abrir el chatbot con cierto canal 

    @param canal  Canal con el que se va a abrir el chatbot (Web o Telegram)
*/
function openChatbot(canal) {
    /* Obtención de la URL del servidor GPU */
    var url = document.getElementById('url').innerText;

    /* Inicialización de una petición HTTPS */
    const Http = new XMLHttpRequest();

    /* Apertura de una petición GET con dirección a la URL del servidor GPU */
    Http.open("GET", url);

    /* Envío de la petición GET */
    Http.send();

    /* Función para gestionar la respuesta a la petición */
    Http.onreadystatechange = (e) => {
        /* Texto que se devuelve como respuesta a la petición */
        var response = Http.responseText;

        /* Si el servidor GPU no está disponible */
        if (response != 'Server GPU ON') {
            /* Se muestra una ventana emergente indicando que no está disponible el servidor GPU */
            Swal.fire({
                icon: 'error',
                title: 'Error de conexión...',
                text: 'El servidor GPU no está disponible en este momento.'
            });
        } else {
            /* En caso contrario */

            /* Se muestra una ventana emergente que pregunta si se quiere deducir la edad a partir de una foto */
            Swal.fire({
                title: '¿Quiere deducir su edad mediante una foto?',
                text: "¡No podrás revertir tu decisión!",
                icon: 'warning',
                showCancelButton: true,
                confirmButtonColor: '#00FF24',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Si',
                cancelButtonText: 'No'
            }).then((result) => {
                /* Si se confirma el permiso para deducir a través de una foto */
                if (result.isConfirmed) {
                    /* Se abre la sección de deducción de edad indicando el canal elegido */
                    openURL('./capture_image', canal);
                } else {
                    /* En caso contrario */

                    /* Se muestra una ventana emergente que pregunta con que rango de edad se asocia el usuario */
                    Swal.fire({
                        title: '¿Cuál es tu edad?',
                        text: 'Debes indicar cual de los siguientes rangos de edad se asocia más con tu edad actual.',
                        showCancelButton: true,
                        confirmButtonColor: '#2563B9',
                        cancelButtonColor: '#EBB241',
                        confirmButtonText: 'Adulto',
                        cancelButtonText: 'Niño'
                    }).then((result) => {
                        /* Si se indica Adulto */
                        if (result.isConfirmed) {
                            /* Si el canal elegido es la Web */
                            if (canal == "web") {
                                /* Se abre la URL hacia la interfaz web del chatbot para adultos */
                                openURL('./' + document.getElementById('web_adult').innerText);
                            } else if (canal == "telegram") {
                                /* Si el canal elegido es Telegram */

                                /* Se abre la URL hacia el chatbot para adultos de Telegram */
                                openURL(document.getElementById('telegram_adult').innerText);
                            }
                        } else {
                            /* Si se indica Niño */

                            /* Si el canal elegido es la Web */
                            if (canal == "web") {
                                /* Se abre la URL hacia la interfaz web del chatbot para niños */
                                openURL('./' + document.getElementById('web_child').innerText);
                            } else if (canal == "telegram") {
                                /* Si el canal elegido es Telegram */

                                /* Se abre la URL hacia el chatbot para niños de Telegram */
                                openURL(document.getElementById('telegram_child').innerText);
                            }
                        }
                    });
                }
            });
        }
    };
}


/* Función para obtener la afirmación o negación de que el objeto se encuentra en estado visible 

    @param element  Elemento de la página web a examinar

    @return Afirmación o negación de que el objeto se encuentra en estado visible
*/
function isVisible(element) {
    return element.offsetWidth > 0 || element.offsetHeight > 0;
}


/* Función para obtener el canvas activo según el dispositivo que se esté usando

    @return Canvas que es visible
*/
function obtenerCanvas() {
    /* Canvas cuando el dispositivo es un portátil */
    var canvasLaptop = document.getElementById('canvasLaptop');

    /* Canvas cuando el dispositivo es una tablet en orientación vertical */
    var canvasTabletVertical = document.getElementById('canvasTabletVertical');

    /* Canvas cuando el dispositivo es una table en orientación horizontal */
    var canvasTabletHorizontal = document.getElementById('canvasTabletHorizontal');

    /* Canvas cuando el dispositivo es un teléfono móvil */
    var canvasMovil = document.getElementById('canvasMovil');

    /* Selección del canvas según la visibilidad de los distintos canvas */
    if (isVisible(canvasLaptop)) {
        return 'canvasLaptop';
    }
    if (isVisible(canvasTabletVertical)) {
        return 'canvasTabletVertical';
    }
    if (isVisible(canvasTabletHorizontal)) {
        return 'canvasTabletHorizontal';
    }
    if (isVisible(canvasMovil)) {
        return 'canvasMovil';
    }
}


/* Función para obtener una foto tomada con la cámara del dispositivo y mostrarla en el canvas */
function snapPhoto() {
    /* Obtención del canvas */
    var canvas = document.getElementById(obtenerCanvas());

    /* Obtención de la zona de dibujado del canvas */
    var context = canvas.getContext('2d');

    /* Obteción del elemento que captura la imagen */
    var video = document.getElementById('video');

    /* Relación entre el ancho y la altura del canvas */
    var relacion = 1.33;

    /* Cálculo del ancho del vídeo */
    var videoHeight = video.clientWidth / relacion;
    /* Cálculo del punto del ancho de la pantalla donde comienza el elemento */
    var inicioHeight = (video.clientHeight - videoHeight) / 2;

    /* Dibujado de la imagen captada en el canvas */
    context.drawImage(video, 0, inicioHeight, video.clientWidth, videoHeight);

    /* Hacer visible el botón para enviar la imagen para su procesado */
    document.getElementById('send_button').style.display = 'block';
}


/* Función para obtener una foto tomada con la cámara del dispositivo y mostrarla en el canvas */
function sendPhoto() {
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
            /* Obtención de la URL del servidor GPU */
            var url = document.getElementById('url').innerText;

            /* Elaboración de la URL completa que se va a abrir */
            url_deduct = url + '/' + document.getElementById('deduct').innerText;

            /* Obtención del canavas que contiene la imagen */
            var canvas = document.getElementById(obtenerCanvas());

            /* Obtención de la imagen en formato base64 */
            var imgBase64 = canvas.toDataURL("image/jpeg", 1.0);

            /* Inicialización de una petición HTTPS */
            const Http = new XMLHttpRequest();

            /* Apertura de una petición POST con dirección a la sección de deducción del servidor GPU */
            Http.open("POST", url_deduct, true);

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
        }
    });
}