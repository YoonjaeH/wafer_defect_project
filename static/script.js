document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const predictBtn = document.getElementById('predictBtn');
    const resultsDiv = document.getElementById('results');
    const predictionText = document.getElementById('predictionText');
    const gradcamImage = document.getElementById('gradcamImage');

    predictBtn.addEventListener('click', async () => {
        if (!imageUpload.files || imageUpload.files.length === 0) {
            alert('Please select an image file first.');
            return;
        }

        const file = imageUpload.files[0];
        const formData = new FormData();
        formData.append('file', file);

        // Show loading state (optional, but good practice)
        predictionText.textContent = 'Classifying...';
        resultsDiv.classList.remove('hidden');
        gradcamImage.src = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();

            // Update the UI with the results
            predictionText.textContent = data.predicted_class;
            gradcamImage.src = data.gradcam_image;

        } catch (error) {
            console.error('Error:', error);
            predictionText.textContent = 'An error occurred.';
        }
    });
});