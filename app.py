from flask import Flask, request, jsonify, render_template
from wafer_cnn import WaferCNN
from utils import transform_image, get_prediction, generate_gradcam, CLASS_NAMES

import torch

app = Flask(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "wafer_defect_model.pth"

model = WaferCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img_bytes = file.read()
        tensor, original_image = transform_image(img_bytes)
        tensor = tensor.to(DEVICE)

        predicted_idx = get_prediction(model, tensor)
        predicted_class = CLASS_NAMES[predicted_idx]
        
        gradcam_image = generate_gradcam(model, tensor, original_image, predicted_idx)

        return jsonify({
            'predicted_class': predicted_class,
            'gradcam_image': gradcam_image
        })

if __name__ == '__main__':
    app.run(debug=True)