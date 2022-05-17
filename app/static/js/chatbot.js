import {
    openChatbot
} from '/static/js/functions.js';



var open_web = document.getElementById("open_web");

open_web.onclick = openChatbot('web');

var open_telegram = document.getElementById("open_telegram");

open_telegram.onclick = openChatbot('telegram');