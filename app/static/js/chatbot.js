/* Botón para ir a la interfaz web del chatbot */
var open_web = document.getElementById("open_web");

/* Definir evento de click para el botón para ir a la interfaz web */
open_web.onclick = function() {
    /* Apertura de la interfaz web del chatbot */
    openChatbot('web');
};

/* Botón para ir a la interfaz de Telegram del chatbot */
var open_telegram = document.getElementById("open_telegram");

/* Definir evento de click para el botón para ir a la interfaz de Telegram */
open_telegram.onclick = function() {
    /* Apertura de la interfaz de Telegram del chatbot */
    openChatbot('telegram');
};

/* Link de la barra de navegación para ir a la página principal */
var link_inicio = document.getElementById("link_inicio");

/* Definir evento de click para el link para ir a la página principal */
link_inicio.onclick = function() {
    /* Ir a la página principal */
    openURL('./');
};

/* Link de la barra de navegación lateral para ir a la página principal */
var link_inicio_lateral = document.getElementById("link_inicio_lateral");

/* Definir evento de click para el link lateral para ir a la página principal */
link_inicio_lateral.onclick = function() {
    /* Ir a la página principal */
    openURL('./');
};