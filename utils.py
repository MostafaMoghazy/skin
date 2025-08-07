import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd

def load_disease_model():
    """Load the disease prediction model with error handling"""
    try:
        # Try .keras format first
        model = load_model("skindisease.keras", compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except:
        try:
            # Try .h5 format
            model = load_model("skindisease.h5", compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            raise Exception(f"Could not load model: {e}")

def predict_disease(model, img_file, class_names):
    """Predict disease from uploaded image"""
    # Load and preprocess image
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence

def scrape_doctors(city, specialty="dermatology"):
    """Return sample doctor data (simplified version)"""
    # For now, return sample data to avoid web scraping issues
    sample_doctors = [
        {"name": "Dr. Ahmed Hassan", "specialty": "Dermatologist", "location": f"{city} Medical Center"},
        {"name": "Dr. Sarah Mohamed", "specialty": "Skin Specialist", "location": f"{city} Clinic"},
        {"name": "Dr. Omar Mahmoud", "specialty": "Dermatology Consultant", "location": f"{city} Hospital"},
    ]
    
    return pd.DataFrame(sample_doctors)
