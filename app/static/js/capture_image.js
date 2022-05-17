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
            };
        } else {
            e.target.value = "";
        }
    });
});



var snap = document.getElementById("snap");

snap.onclick = function() {
    snapPhoto();
};

var send = document.getElementById("send");

send.onclick = function() {
    sendPhoto();
};