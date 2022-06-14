/* Switch para el cambio a modo oscuro */
const buttonSwitch = document.querySelector('#switch_dark_mode');

/* Definir evento de click para el switch */
buttonSwitch.addEventListener('click', () => {
    /* Añadir la etiqueta dark_mode a todos los elementos de la página web, y en caso de que ya estemos en modo oscuro se elimina esta etiqueta de todos los elementos */
    document.body.classList.toggle('dark_mode');
    /* Añadir la etiqueta active al switch, y en caso de que ya estemos en modo oscuro se elimina esta etiqueta al switch */
    buttonSwitch.classList.toggle('active');
});

/* Botón para desplegar la barra de navegación lateral */
const buttonSidebar = document.querySelector('#button_sidebar');
/* Barra de navegación lateral */
const sidebar = document.querySelector('#sidebar');

/* Definir evento de click para el botón para desplegar la barra de navegación lateral */
buttonSidebar.addEventListener('click', () => {
    /* Añadir la etiqueta deploy al botón, y en caso de que ya estemos en modo oscuro se elimina esta etiqueta al botón */
    sidebar.classList.toggle('deploy');
});

/* Definir un evento de cambio de tamaño para la ventana */
window.addEventListener('resize', () => {
    /* Si la barra de navegación laterla está desplegada */
    if (sidebar.classList.contains('deploy'))
    /* Si el tamaño del ancho de la ventana supera los 750 píxeles */
        if (document.documentElement.clientWidth > 750)
        /* Eliminar etiqueta deploy a la barra de navegación lateral */
            sidebar.classList.toggle('deploy');
});

/* Creación de un evento de click */
let event = new Event("click");

/* Si el switch de cambio a modo oscuro estaba activado en la anterior ventana */
if (document.getElementById('dark_mode').innerText == "switch_dark_mode active") {
    /* Se aplica el evento de click al switch para pasar a modo oscuro y mantener el estado al cambiar de ventana */
    buttonSwitch.dispatchEvent(event);
}