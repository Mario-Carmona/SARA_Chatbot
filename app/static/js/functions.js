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

    /* Inicialización de una petición HTTP */
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


function isVisible(element) {
    return element.offsetWidth > 0 || element.offsetHeight > 0;
}

function obtenerCanvas() {
    var canvasLaptop = document.getElementById('canvasLaptop');
    var canvasTabletVertical = document.getElementById('canvasTabletVertical');
    var canvasTabletHorizontal = document.getElementById('canvasTabletHorizontal');
    var canvasMovil = document.getElementById('canvasMovil');

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

function snapPhoto() {
    var canvas = document.getElementById(obtenerCanvas());
    var context = canvas.getContext('2d');
    var video = document.getElementById('video');
    var relacion = 1.33;

    var videoHeight = video.clientWidth / relacion;
    var inicioHeight = (video.clientHeight - videoHeight) / 2;
    context.drawImage(video, 0, inicioHeight, video.clientWidth, videoHeight);

    document.getElementById('send_button').style.display = 'block';
}

function sendPhoto() {
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
        if (result.isConfirmed) {
            var url = document.getElementById('url').innerText;

            url_deduct = url + '/' + document.getElementById('deduct').innerText;

            var canvas = document.getElementById(obtenerCanvas());
            var imgBase64 = canvas.toDataURL("image/jpeg", 1.0);

            console.log(imgBase64)

            const Http = new XMLHttpRequest();
            Http.open("POST", url_deduct, true);
            Http.setRequestHeader("Content-Type", "application/json");

            Http.onreadystatechange = function() {
                var age = Http.responseText
                console.log(age)

                if (document.getElementById('canal').innerText == "web") {
                    openURL('./' + document.getElementById('web_' + age).innerText);
                } else if (document.getElementById('canal').innerText == "telegram") {
                    openURL(document.getElementById('telegram_' + age).innerText);
                }
            };

            var data = {
                imagen: imgBase64
            }

            Http.send(JSON.stringify(data));

            var loading = document.getElementById('loading');
            loading.classList.toggle('active');
        }
    });
}