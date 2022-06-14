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

/* Link de la barra de navegación para ir a la sección de actividad chatbot */
var link_chatbot = document.getElementById("link_chatbot");

/* Definir evento de click para el link para ir a la sección de actividad chatbot */
link_chatbot.onclick = function() {
    /* Ir a la sección de actividad chatbot */
    openURL('./chatbot');
};

/* Link de la barra de navegación lateral para ir a la sección de actividad chatbot */
var link_chatbot_lateral = document.getElementById("link_chatbot_lateral");

/* Definir evento de click para el link lateral para ir a la sección de actividad chatbot */
link_chatbot_lateral.onclick = function() {
    /* Ir a la sección de actividad chatbot */
    openURL('./chatbot');
};