function obtain_status_dark_mode() {
    var buttonSwitch = document.querySelector('#switch_dark_mode');
    return buttonSwitch.classList.value;
}

function openURL(url, canal = '') {
    var url_completa = '';
    if (canal == '') {
        url_completa = url + '?dark_mode=' + obtain_status_dark_mode();
    } else {
        url_completa = url + '?canal=' + canal + '&dark_mode=' + obtain_status_dark_mode();
    }
    window.open(url_completa, "_self");
}


function openChatbot(canal) {
    var url = document.getElementById('url').innerText;

    const Http = new XMLHttpRequest();
    Http.open("GET", url);

    Http.send();

    Http.onreadystatechange = (e) => {
        var response = Http.responseText;

        console.log(Http);

        console.log(response);

        if (response != 'Server GPU ON') {
            Swal.fire({
                icon: 'error',
                title: 'Error de conexión...',
                text: 'El servidor GPU no está disponible en este momento.'
            });
        } else {
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
                if (result.isConfirmed) {
                    openURL('./capture_image', canal);
                } else {
                    Swal.fire({
                        title: '¿Cuál es tu edad?',
                        text: 'Debes indicar cual de los siguientes rangos de edad se asocia más con tu edad actual.',
                        showCancelButton: true,
                        confirmButtonColor: '#2563B9',
                        cancelButtonColor: '#EBB241',
                        confirmButtonText: 'Adulto',
                        cancelButtonText: 'Niño'
                    }).then((result) => {
                        if (result.isConfirmed) {
                            if (canal == "web") {
                                openURL('./' + document.getElementById('web_adult').innerText);
                            } else if (canal == "telegram") {
                                openURL(document.getElementById('telegram_adult').innerText);
                            }
                        } else {
                            if (canal == "web") {
                                openURL('./' + document.getElementById('web_child').innerText);
                            } else if (canal == "telegram") {
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