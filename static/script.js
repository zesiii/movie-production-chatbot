function sendMessage() {
    var input = document.getElementById("chatInput");
    var message = input.value.trim();

    if (message) {
        displayMessage(message, "user");

        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (message.startsWith("Movie Content:")) {
                var prediction = "Prediction: $" + parseFloat(data.response.prediction_output).toFixed(2);
                var report = formatMovieReport(data.response.movie_report);

                displayMessage(prediction, "bot");
                displayMessage(report, "bot");
            } else {
                displayMessage(data.response, "bot");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage("Error: Unable to fetch data", "bot");
        });

        input.value = "";
    }
}

function formatMovieReport(report) {
    // Split the report into lines and process them
    var lines = report.split('\n');
    var formattedReport = "";

    lines.forEach(line => {
        if (line.startsWith('Content Summary:')) {
            formattedReport += line.replace('Movie Concept:', '').trim() + '\n';
        } else if (line.startsWith('Cast Recommendation:')) {
            // Keep only the first five actors and their reasons
            formattedReport += "Cast Recommendations:\n";
        } else if (!line.startsWith('Movie Content:') && !line.startsWith('Genre:') && !line.includes('Budget Recommendation:')) {
            formattedReport += line + '\n';
        }
    });

    return formattedReport.trim();
}

// Existing displayMessage function remains the same


// Existing displayMessage function




function displayMessage(message, sender) {
    var chatBox = document.getElementById("chatBox");

    // Split the message into paragraphs for better readability
    var paragraphs = message.split('\n').filter(p => p.trim() !== '');
    paragraphs.forEach(paragraph => {
        var messageElement = document.createElement("div");
        messageElement.classList.add("chat-message", sender);

        var text = document.createElement("span");

        // Making specific parts bold
        paragraph = paragraph.replace(/(Prediction:|Content Summary:|Reference Movies:|Cast Recommendations:)/g, '<strong>$1</strong>');

        // Set innerHTML to include HTML tags
        text.innerHTML = paragraph;

        if (sender === "user") {
            // User message: append text first, then the user icon
            messageElement.appendChild(text);
            var userImg = document.createElement("img");
            userImg.src = staticBaseUrl + 'chat_client.png';
            userImg.classList.add("chat-icon");
            messageElement.appendChild(userImg);
        } else {
            // Bot response: append the KPMG icon first, then text
            var kpmgImg = document.createElement("img");
            kpmgImg.src = staticBaseUrl + 'chat_kpmg.png';
            kpmgImg.classList.add("chat-icon");
            messageElement.appendChild(kpmgImg);
            messageElement.appendChild(text);
        }

        chatBox.appendChild(messageElement);
    });
    chatBox.scrollTop = chatBox.scrollHeight;
}





fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: 'Movie Content: your movie content'})
})
.then(response => response.json())
.then(data => {
    console.log('Received data:', data); // Log the data to inspect its structure

    // Access and display the prediction_output
    if (data.response && data.response.prediction_output !== undefined) {
        document.getElementById('outputElementId').textContent = data.response.prediction_output;
    } else {
        // Handle other types of responses or errors
        document.getElementById('outputElementId').textContent = 'No prediction output received';
    }
})
.catch(error => console.error('Error:', error));
