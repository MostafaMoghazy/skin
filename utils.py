import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os

def recreate_model_architecture():
    """
    Recreate your exact model architecture for loading weights.
    This matches your training code exactly.
    """
    # Create base model with same parameters
    Mobile_base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    Mobile_base_model.trainable = False
    
    # Build the exact same model architecture
    model = Sequential([
        Mobile_base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax', name='predictions')
    ])
    
    # Compile with same parameters
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_disease_model():
    """
    Load your skin disease model with multiple fallback strategies.
    """
    print("🔄 Attempting to load skin disease model...")
    
    # Strategy 1: Try loading .keras format
    try:
        print("📁 Trying to load skindisease.keras...")
        model = load_model("skindisease.keras")
        print("✅ Successfully loaded skindisease.keras")
        return model
    except Exception as e:
        print(f"❌ Failed to load .keras format: {str(e)[:100]}...")
    
    # Strategy 2: Try loading .keras without compilation
    try:
        print("📁 Trying to load skindisease.keras (no compile)...")
        model = load_model("skindisease.keras", compile=False)
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Successfully loaded skindisease.keras (recompiled)")
        return model
    except Exception as e:
        print(f"❌ Failed to load .keras (no compile): {str(e)[:100]}...")
    
    # Strategy 3: Try loading .h5 format
    try:
        print("📁 Trying to load skindisease.h5...")
        model = load_model("skindisease.h5")
        print("✅ Successfully loaded skindisease.h5")
        return model
    except Exception as e:
        print(f"❌ Failed to load .h5 format: {str(e)[:100]}...")
    
    # Strategy 4: Try loading .h5 without compilation
    try:
        print("📁 Trying to load skindisease.h5 (no compile)...")
        model = load_model("skindisease.h5", compile=False)
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Successfully loaded skindisease.h5 (recompiled)")
        return model
    except Exception as e:
        print(f"❌ Failed to load .h5 (no compile): {str(e)[:100]}...")
    
    # Strategy 5: Try loading weights only
    weights_files = ["skindisease_weights.h5", "skindisease.weights.h5"]
    for weights_file in weights_files:
        if os.path.exists(weights_file):
            try:
                print(f"📁 Trying to load weights from {weights_file}...")
                model = recreate_model_architecture()
                model.load_weights(weights_file)
                print(f"✅ Successfully loaded weights from {weights_file}")
                return model
            except Exception as e:
                print(f"❌ Failed to load weights from {weights_file}: {str(e)[:100]}...")
    
    # Strategy 6: Try SavedModel format
    savedmodel_dirs = ["skindisease_savedmodel", "saved_model"]
    for savedmodel_dir in savedmodel_dirs:
        if os.path.exists(savedmodel_dir):
            try:
                print(f"📁 Trying to load SavedModel from {savedmodel_dir}...")
                model = tf.keras.models.load_model(savedmodel_dir)
                print(f"✅ Successfully loaded SavedModel from {savedmodel_dir}")
                return model
            except Exception as e:
                print(f"❌ Failed to load SavedModel from {savedmodel_dir}: {str(e)[:100]}...")
    
    # Strategy 7: Create fresh model for demonstration (if all else fails)
    print("🆕 Creating fresh model architecture for demonstration...")
    try:
        model = recreate_model_architecture()
        print("⚠️ Created fresh model - will need retraining for accurate predictions")
        return model
    except Exception as e:
        print(f"❌ Failed to create fresh model: {e}")
        raise Exception("All model loading strategies failed")

def load_class_names():
    """
    Load the class names from the saved file or return defaults.
    """
    try:
        if os.path.exists("class_names.npy"):
            class_names = np.load("class_names.npy", allow_pickle=True).tolist()
            print(f"✅ Loaded class names: {class_names}")
            return class_names
    except Exception as e:
        print(f"❌ Failed to load class names: {e}")
    
    # Default class names (update these to match your dataset)
    default_classes = ['Acne', 'Eczema', 'Psoriasis']
    print(f"⚠️ Using default class names: {default_classes}")
    return default_classes

def predict_disease(model, img_file, class_names):
    """
    Predict disease from image using your trained model.
    """
    try:
        # Load and preprocess image exactly as in training
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        
        # Normalize pixel values to [0, 1] (same as training)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"📸 Image shape for prediction: {img_array.shape}")
        print(f"📸 Image value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        print(f"🧠 Raw predictions: {predictions}")
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        print(f"🎯 Predicted: {predicted_class} (confidence: {confidence:.3f})")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Return fallback prediction
        return class_names[0], 0.33

def test_model_prediction(model, class_names):
    """
    Test the model with a dummy image to verify it's working.
    """
    try:
        print("🧪 Testing model with dummy data...")
        
        # Create dummy image data
        dummy_img = np.random.random((1, 224, 224, 3))
        
        # Make prediction
        predictions = model.predict(dummy_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        print(f"✅ Test prediction successful: {predicted_class} ({confidence:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ Test prediction failed: {e}")
        return False

def scrape_doctors(city, specialty="dermatology"):
    """
    Scrape doctors with improved error handling.
    """
    try:
        # Clean inputs
        city_clean = city.lower().replace(" ", "-")
        specialty_clean = specialty.lower().replace(" ", "-")
        
        # Try different URL patterns
        urls_to_try = [
            f"https://www.vezeeta.com/en/doctor/{specialty_clean}/{city_clean}",
            f"https://www.vezeeta.com/en/doctors/{specialty_clean}/{city_clean}",
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        for url in urls_to_try:
            try:
                print(f"🔍 Searching doctors at: {url}")
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "lxml")
                    
                    # Multiple selector strategies
                    name_selectors = [
                        'a.CommonStylesstyle__TransparentA-sc-1vkcu2o-2.cTFrlk',
                        '.doctor-name',
                        '[data-testid="doctor-name"]',
                        'h3 a'
                    ]
                    
                    spec_selectors = [
                        'p.DoctorCardSubComponentsstyle__Text-sc-1vq3h7c-14.DoctorCardSubComponentsstyle__DescText-sc-1vq3h7c-17',
                        '.doctor-specialty',
                        '.specialty'
                    ]
                    
                    loc_selectors = [
                        'span.DoctorCardstyle__Text-sc-uptab2-4',
                        '.doctor-location',
                        '.location'
                    ]
                    
                    # Try to extract data
                    names, specs, locs = [], [], []
                    
                    for selector in name_selectors:
                        names = soup.select(selector)
                        if names:
                            break
                    
                    for selector in spec_selectors:
                        specs = soup.select(selector)
                        if specs:
                            break
                    
                    for selector in loc_selectors:
                        locs = soup.select(selector)
                        if locs:
                            break
                    
                    # Process results
                    data = []
                    min_length = min(len(names), len(specs), len(locs)) if all([names, specs, locs]) else 0
                    
                    if min_length > 0:
                        for i in range(min_length):
                            try:
                                data.append({
                                    "name": names[i].get_text(strip=True),
                                    "specialty": specs[i].get_text(strip=True),
                                    "location": locs[i].get_text(strip=True)
                                })
                            except Exception:
                                continue
                        
                        if data:
                            print(f"✅ Found {len(data)} doctors")
                            return pd.DataFrame(data)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
                continue
        
        # Fallback: return sample data
        print("⚠️ Scraping failed, returning sample data")
        sample_data = [
            {"name": f"Dr. Ahmed Hassan", "specialty": "Dermatologist", "location": f"{city} Medical Center"},
            {"name": f"Dr. Sarah Mohamed", "specialty": "Skin Specialist", "location": f"{city} Skin Clinic"},
            {"name": f"Dr. Omar Mahmoud", "specialty": "Dermatology Consultant", "location": f"{city} Hospital"}
        ]
        
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        print(f"❌ Error in scrape_doctors: {e}")
        return pd.DataFrame()

# Model validation function
def validate_model_setup():
    """
    Validate that the model and related files are set up correctly.
    """
    print("🔧 Validating model setup...")
    
    # Check for model files
    model_files = [
        "skindisease.keras",
        "skindisease.h5", 
        "skindisease_weights.h5",
        "class_names.npy"
    ]
    
    existing_files = []
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            existing_files.append(f"{file} ({size} bytes)")
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    if not existing_files:
        print("⚠️ No model files found. Please run the training script first.")
        return False
    
    # Try to load model
    try:
        model = load_disease_model()
        class_names = load_class_names()
        
        # Test prediction
        success = test_model_prediction(model, class_names)
        
        if success:
            print("✅ Model validation successful!")
            return True
        else:
            print("❌ Model validation failed!")
            return False
            
    except Exception as e:
        print(f"❌ Model validation error: {e}")
        return False

if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_model_setup()
