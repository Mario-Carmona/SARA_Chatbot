const buttonSwitch = document.querySelector('#switch_dark_mode');

buttonSwitch.addEventListener('click', () => {
    document.body.classList.toggle('dark_mode');
    buttonSwitch.classList.toggle('active');
});