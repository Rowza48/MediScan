document.addEventListener('DOMContentLoaded', () => {
    // Handles file upload and triggers prediction
    const fileInput = document.getElementById('file-upload');
    if (fileInput) {
        fileInput.addEventListener('change', async function(e) {
            const file = e.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);

            // Preview uploaded image
            const reader = new FileReader();
            reader.onload = function(event) {
                document.getElementById('result-img').src = event.target.result;
            };
            reader.readAsDataURL(file);

            // Call backend predict API
            const resultCard = document.getElementById('result-area');
            resultCard.classList.remove('show');
            fetch('/predict', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.result) {
                        document.getElementById('result-label').innerText = data.result;
                        resultCard.classList.add('show');
                    } else {
                        document.getElementById('result-label').innerText = "Error: " + (data.error || 'Unknown error');
                    }
                })
                .catch(() => {
                    document.getElementById('result-label').innerText = "Prediction failed. Please try again.";
                });
        });
    }
});
