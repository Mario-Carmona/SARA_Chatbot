/* Botón de acceso a la actividad chatbot */
var open_chatbot = document.getElementById("open_chatbot");

/* Definir evento de click al botón open_chatbot */
open_chatbot.onclick = function() {
    /* Abrir actividad chatbot */
    openURL('./chatbot');
};