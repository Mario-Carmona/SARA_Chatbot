const buttonSwitch = document.querySelector('#switch_dark_mode');

buttonSwitch.addEventListener('click', () => {
    document.body.classList.toggle('dark_mode');
    buttonSwitch.classList.toggle('active');

    obtain_status_dark_mode();
});

const buttonSidebar = document.querySelector('#button_sidebar');
const sidebar = document.querySelector('#sidebar');

buttonSidebar.addEventListener('click', () => {
    sidebar.classList.toggle('deploy');
});

window.addEventListener('resize', () => {
    if (sidebar.classList.contains('deploy'))
        if (document.documentElement.clientWidth > 750)
            sidebar.classList.toggle('deploy');
});



function obtain_status_dark_mode() {
    var buttonSwitch = document.querySelector('#switch_dark_mode');
    console.log(buttonSwitch.classList.value)
}


/********************************/




obtain_status_dark_mode();

// Grab elements, create settings, etc.
var video = document.getElementById('video');

// Get access to the camera!
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        //video.src = window.URL.createObjectURL(stream);
        video.srcObject = stream;
        video.play();
    });
}




function getBase64(file) {
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function() {
        var url = document.getElementById('url').innerText;

        url = url + '/deduct';

        var imgBase64 = reader.result;

        console.log(imgBase64)

        const Http = new XMLHttpRequest();
        Http.open("POST", url, true);
        Http.setRequestHeader("Content-Type", "application/json");

        Http.onreadystatechange = function() {
            var age = Http.responseText
            console.log(age)

            if (document.getElementById('canal').innerText == "web") {
                window.location.replace('./' + document.getElementById('web_' + age).innerText);
            } else if (document.getElementById('canal').innerText == "telegram") {
                window.location.replace(document.getElementById('telegram_' + age).innerText)
            }
        };

        var data = {
            imagen: imgBase64
        }

        Http.send(JSON.stringify(data));

        var loading = document.getElementById('loading');
        loading.classList.toggle('active');
    };
}



const camera = document.querySelector('#camera');
camera.addEventListener('change', function(e) {
    //photo.src = URL.createObjectURL(e.target.files[0]);
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
            var reader = new FileReader();
            reader.readAsDataURL(e.target.files[0]);
            reader.onload = function() {
                var url = document.getElementById('url').innerText;

                url = url + '/deduct';

                var imgBase64 = reader.result;

                console.log(imgBase64)

                const Http = new XMLHttpRequest();
                Http.open("POST", url, true);
                Http.setRequestHeader("Content-Type", "application/json");

                Http.onreadystatechange = function() {
                    var age = Http.responseText
                    console.log(age)

                    if (document.getElementById('canal').innerText == "web") {
                        window.location.replace('./' + document.getElementById('web_' + age).innerText);
                    } else if (document.getElementById('canal').innerText == "telegram") {
                        window.location.replace(document.getElementById('telegram_' + age).innerText)
                    }
                };

                var data = {
                    imagen: imgBase64
                }

                Http.send(JSON.stringify(data));

                var loading = document.getElementById('loading');
                loading.classList.toggle('active');
            };
        } else {
            e.target.value = "";
        }
    });
});



/*********************/



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
                    window.location.replace('./' + document.getElementById('web_' + age).innerText);
                } else if (document.getElementById('canal').innerText == "telegram") {
                    window.location.replace(document.getElementById('telegram_' + age).innerText)
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