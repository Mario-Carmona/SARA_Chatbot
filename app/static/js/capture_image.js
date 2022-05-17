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
    console.log(buttonSwitch.classList)
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