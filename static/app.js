var typed = true;

$(document).ready(async function () {
    $('.indicator').hide()

    $('#inputText').on('input', async (e) => {
        if (e.target.value.length) {
            $('.indicator').show()
            typed = true;
        } else {
            $('.indicator').hide()
            $("#languageClass").text("N/A");
            $("#outputText").text();
        }

        if (e.target.value.length % 2 == 0 && e.target.value.length) {
            await fetch('http://127.0.0.1:5000/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: e.target.value })
            })
            .then(response => response.json())
            .then(data => {
                $("#languageClass").text(data.language);
                $("#outputText").text(data.translated);
            })
            .catch(error => console.error('Error:', error));
        }
    })

    setInterval(function() {
        if (!typed) {
            $('.indicator').hide();
        }

        typed = false;
    }, 1000);
})