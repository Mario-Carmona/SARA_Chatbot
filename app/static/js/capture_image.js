const buttonSwitch = document.querySelector('#switch_dark_mode');

buttonSwitch.addEventListener('click', () => {
    document.body.classList.toggle('dark_mode');
    buttonSwitch.classList.toggle('active');
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




/********************************/

/*
'use strict';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById("snap");
const errorMsgElement = document.querySelector('span#errorMsg');
*/

/*
navigator.mediaDevices.getUserMedia({ video: { width: video.clientWidth, height: video.clientHeight } }).then(function(stream) {
    //video.src = window.URL.createObjectURL(stream);
    video.srcObject = stream;
    video.play();
});
var context = canvas.getContext('2d');
snap.addEventListener("click", function() {
    context.drawImage(video, 0, 0);
});
*/

/*
const constraints = {
    video: {
        width: video.clientWidth,
        height: video.clientHeight
    }
};

// Access webcam
async function init() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        handleSuccess(stream);
    } catch (e) {
        errorMsgElement.innerHTML = `navigator.getUserMedia error:${e.toString()}`;
    }
}



// Success
function handleSuccess(stream) {
    window.stream = stream;
    video.srcObject = stream;
}

// Load init
init();

// Draw image
var context = canvas.getContext('2d');
snap.addEventListener("click", function() {
    context.drawImage(stream, 0, 0);
});
*/



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

var canvas = document.getElementById('canvasLaptop');
var context = canvas.getContext('2d');
var video = document.getElementById('video');
var relacion = 1.33;

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
    var videoHeight = video.clientWidth / relacion;
    var inicioHeight = (video.clientHeight - videoHeight) / 2;
    context.drawImage(video, 0, inicioHeight, video.clientWidth, videoHeight);
});




document.getElementById("send").addEventListener("click", function() {
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
            var canvas = document.getElementById('canvasLaptop');
            var imgBase64 = canvas.toDataURL("image/jpeg", 1.0);

            var url = document.getElementById('url').innerText;


            console.log(typeof imgBase64);

            var url_param = url + "?imagen=" + imgBase64;

            var age;

            $.get(url_param, function(data, status) {
                age = data;
            });

            console.log(age)



            /* 
            var otherParam = {
                headers: {
                    "content-type": "application/json; charset=UTF-8"
                },
                body: Data,
                method: "POST"
            };

            fetch(url, otherParam)
                .then(response => (console.log(response)))
                .then(data => (console.log(data)))
            */



            /*

            // Example POST method implementation:
            async function postData(url = '', data = {}) {
                // Default options are marked with *
                const response = await fetch(url, {
                    method: 'POST', // *GET, POST, PUT, DELETE, etc.
                    mode: 'cors', // no-cors, *cors, same-origin
                    headers: {
                        'Content-Type': 'application/json'
                            // 'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: JSON.stringify(data) // body data type must match "Content-Type" header
                });
                return response.json(); // parses JSON response into native JavaScript objects
            }

            postData(url, data)
                .then(data => {
                    console.log(data); // JSON data parsed by `data.json()` call
                });

            */









            /*
            $.post(url, data, function(data, status) {
                console.log(`${data} and status is ${status}`)
            });
            */

            //window.location.replace("./interface");
        }
    });
});