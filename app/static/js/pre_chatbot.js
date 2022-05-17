import { openURL } from '/static/js/base.js';

function openChatbot(canal) {
    var url = document.getElementById('url').innerText;

    if (url == '') {
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
}