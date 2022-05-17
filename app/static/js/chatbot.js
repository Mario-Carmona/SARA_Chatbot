var open_web = document.getElementById("open_web");

open_web.onclick = function() {
    openChatbot('web');
};

var open_telegram = document.getElementById("open_telegram");

open_telegram.onclick = function() {
    openChatbot('telegram');
};

var link_inicio = document.getElementById("link_inicio");

link_inicio.onclick = function() {
    openURL('./');
};