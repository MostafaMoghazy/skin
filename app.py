import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import traceback

# Page config
st.set_page_config(page_title="Disease Predictor", layout="wide")

st.title("🩺 AI-Powered Disease Prediction")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Simple model loading function
@st.cache_resource
def load_disease_model():
    """Load the disease prediction model"""
    try:
        # Try different loading methods
        try:
            model = load_model("skindisease.keras", compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except:
            model = load_model("skindisease.h5", compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Simple prediction function
def predict_disease(model, img_file, class_names):
    """Predict disease from image"""
    try:
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Unknown", 0.0

# Simple doctor finder (sample data)
def get_sample_doctors(city):
    """Return sample doctor data"""
    doctors = [
        {"name": "Dr. Ahmed Hassan", "specialty": "Dermatologist", "location": f"{city} Medical Center"},
        {"name": "Dr. Sarah Mohamed", "specialty": "Skin Specialist", "location": f"{city} Clinic"},
        {"name": "Dr. Omar Mahmoud", "specialty": "Dermatology", "location": f"{city} Hospital"}
    ]
    return pd.DataFrame(doctors)

# Main app
def main():
    # Class names - UPDATE THESE to match your dataset!
    class_names = ['Acne', 'Eczema', 'Psoriasis']
    
    # Sidebar
    st.sidebar.header("Upload Your Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    st.sidebar.header("Enter Your City")
    user_city = st.sidebar.text_input("City", value="Cairo")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            try:
                model = load_disease_model()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Model loading error: {e}")
                st.code(traceback.format_exc())
                st.stop()
    
    # Show image
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Image")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            if st.button("🔍 Analyze Image"):
                if st.session_state.model is not None:
                    with st.spinner("Analyzing..."):
                        try:
                            prediction, confidence = predict_disease(
                                st.session_state.model, 
                                uploaded_file, 
                                class_names
                            )
                            
                            st.success(f"**Predicted Disease:** {prediction}")
                            st.info(f"**Confidence:** {confidence*100:.1f}%")
                            
                            # Progress bar for confidence
                            st.progress(confidence)
                            
                            # Show doctors
                            st.subheader("🏥 Recommended Doctors")
                            doctor_df = get_sample_doctors(user_city)
                            
                            for _, doctor in doctor_df.iterrows():
                                st.markdown(f"""
                                **🧑‍⚕️ {doctor['name']}**  
                                Specialty: {doctor['specialty']}  
                                Location: 📍 {doctor['location']}
                                ---
                                """)
                                
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            st.code(traceback.format_exc())
                else:
                    st.error("Model not loaded!")
    else:
        st.info("👆 Please upload an image to begin analysis")
        
        # Show some info while waiting
        st.markdown("""
        ### How to use this tool:
        1. **Upload an image** of the affected skin area
        2. **Enter your city** for doctor recommendations  
        3. **Click Analyze** to get AI predictions
        
        ### Supported conditions:
        - **Acne**: Common inflammatory skin condition
        - **Eczema**: Chronic inflammatory skin disorder
        - **Psoriasis**: Autoimmune skin condition
        
        ⚠️ **Important**: This is for educational purposes only. Always consult a healthcare professional for medical advice.
        """)

# Run the app
if __name__ == "__main__":
    main()
