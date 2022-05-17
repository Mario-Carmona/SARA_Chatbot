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


let event = new Event("click");

try {
    if (document.getElementById('dark_mode').innerText == "switch_dark_mode active") {
        buttonSwitch.dispatchEvent(event);
    }
} catch (e) {}


function obtain_status_dark_mode() {
    var buttonSwitch = document.querySelector('#switch_dark_mode');
    return buttonSwitch.classList.value;
}