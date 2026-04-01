import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import datetime

app = Flask(__name__)

# 1. Load the model globally for speed
model = tf.keras.models.load_model('skin_model.h5')
classes = ['Acne', 'Dark Circles', 'Dryness', 'Pigmentation', 'Wrinkles']

# 2. Clinical Data Mapping
analysis_data = {
    'Acne': {'routine': 'Foaming Salicylic acid cleanser twice daily.', 'precautions': 'Avoid physical extraction; limit dairy.', 'formulations': 'Adapalene 0.1% or Benzoyl Peroxide.'},
    'Dark Circles': {'routine': 'Caffeine-infused eye serums; cold compress.', 'precautions': '8h sleep; reduce digital eye strain.', 'formulations': 'Topical Vitamin K or Retinol complexes.'},
    'Dryness': {'routine': 'Creamy, soap-free cleanser; damp-skin hydration.', 'precautions': 'Avoid hot water and alcohol-based toners.', 'formulations': 'Ceramides, Glycerin, or 10% Urea.'},
    'Pigmentation': {'routine': 'Vitamin C (AM), Niacinamide (PM).', 'precautions': 'Mandatory SPF 50; avoid physical scrubs.', 'formulations': 'Alpha Arbutin or Kojic Acid formulations.'},
    'Wrinkles': {'routine': 'Peptide-rich serums; high sun protection.', 'precautions': 'Systemic hydration; avoid smoking.', 'formulations': 'Retinoids or Copper Peptide complexes.'}
}

@app.route("/")
def home():
    return render_template("index.html", r=None)

@app.route("/analyze", methods=['POST'])
def analyze():
    if 'file' not in request.files: return "No file uploaded"
    f = request.files['file']
    
    # Save temporarily to process
    os.makedirs('static/uploads', exist_ok=True)
    path = os.path.join('static/uploads', f.filename)
    f.save(path)
    
    # AI Prediction Logic
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds)
    conf = float(preds[0][idx]) * 100
    
    # Dynamic Severity
    sev = "Acute" if conf > 88 else "Moderate" if conf > 65 else "Mild"
    
    report = {
        'label': classes[idx], 
        'conf': round(conf, 2), 
        'severity': sev, 
        'ts': datetime.datetime.now().strftime("%H%M%S")
    }
    
    return render_template("index.html", r=report, d=analysis_data[classes[idx]])

if __name__ == "__main__":
    # Render provides the PORT as an environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)