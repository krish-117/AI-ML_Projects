from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image
import numpy as np
import io

# Initialize Flask app
app = Flask(__name__)

# Load model, tokenizer, and feature extractor
model = load_model('model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
feature_extractor = load_model('feature_extractor.keras')

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size based on your feature extractor
    image = np.array(image) / 255.0   # Normalization
    image = np.expand_dims(image, axis=0)
    return image

# Caption generation function
def generate_caption(image):
    features = feature_extractor.predict(preprocess_image(image))
    caption = 'startseq'
    for _ in range(34):  # Max caption length
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=34)
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        caption += ' ' + word
        if word == 'endseq':
            break
    return caption.replace('startseq', '').replace('endseq', '')

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# API route to generate captions
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))

    caption = generate_caption(image)
    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
