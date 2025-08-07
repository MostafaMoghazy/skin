import streamlit as st
import sys
import traceback
import numpy as np
from utils import load_disease_model, load_class_names, predict_disease, scrape_doctors, validate_model_setup

# Configure Streamlit page
st.set_page_config(
    page_title="AI Skin Disease Predictor", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# AI Skin Disease Predictor\nBuilt with MobileNetV2 and TensorFlow!"
    }
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 30px;
    font-size: 2.5rem;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-weight: bold;
    background-color: #1f77b4;
    color: white;
    border: none;
}
.stButton > button:hover {
    background-color: #1565c0;
    color: white;
}
.doctor-card {
    border: 2px solid #e0e0e0;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.prediction-card {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #1976d2;
    margin: 20px 0;
}
.confidence-high { color: #2e7d32; font-weight: bold; }
.confidence-medium { color: #f57c00; font-weight: bold; }
.confidence-low { color: #d32f2f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🩺 AI Skin Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by MobileNetV2 Deep Learning</p>', unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_status' not in st.session_state:
        st.session_state.model_status = "not_loaded"

initialize_session_state()

# Model loading section
def load_model_interface():
    if not st.session_state.model_loaded:
        st.info("🤖 Initializing AI model... This may take a moment on first load.")
        
        # Show loading progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Validate setup
            status_text.text("🔍 Validating model files...")
            progress_bar.progress(20)
            
            # Step 2: Load model
            status_text.text("🧠 Loading AI model...")
            progress_bar.progress(50)
            model = load_disease_model()
            
            # Step 3: Load class names
            status_text.text("📚 Loading class information...")
            progress_bar.progress(70)
            class_names = load_class_names()
            
            # Step 4: Final validation
            status_text.text("✅ Finalizing setup...")
            progress_bar.progress(100)
            
            # Update session state
            st.session_state.model = model
            st.session_state.class_names = class_names
            st.session_state.model_loaded = True
            st.session_state.model_status = "loaded"
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎉 AI model loaded successfully!")
            st.balloons()
            
            # Show model info
            with st.expander("🔍 Model Information", expanded=False):
                st.write(f"**Architecture**: MobileNetV2 + Custom Classification Head")
                st.write(f"**Classes**: {', '.join(class_names)}")
                st.write(f"**Input Size**: 224x224 pixels")
                st.write(f"**Model Status**: Ready for predictions ✅")
            
            st.rerun()
            
        except Exception as e:
            st.session_state.model_status = "error"
            st.error(f"❌ Failed to load AI model: {str(e)}")
            
            # Detailed troubleshooting
            with st.expander("🔧 Detailed Error Information"):
                st.write("**Error Details:**")
                st.code(str(e))
                
                st.write("**Common Solutions:**")
                st.write("1. **Re-run your training script** to create fresh model files")
                st.write("2. **Check file paths** - ensure model files are in the correct directory")
                st.write("3. **Verify TensorFlow version** compatibility")
                st.write("4. **Clear any corrupted files** and retrain if necessary")
                
                st.write("**Expected Files:**")
                st.code("""
skindisease.keras          # Main model file
skindisease.h5            # Alternative format
skindisease_weights.h5    # Weights only
class_names.npy           # Class labels
                """)
                
                st.write("**Full Error Traceback:**")
                st.code(traceback.format_exc())
            
            # Retry button
            if st.button("🔄 Retry Loading Model"):
                st.session_state.model_loaded = False
                st.session_state.model_status = "not_loaded"
                st.rerun()

# Main application
def main_application():
    if st.session_state.model_loaded and st.session_state.model is not None:
        
        # Sidebar controls
        st.sidebar.markdown("### 📸 Step 1: Upload Medical Image")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a skin image", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit image of the affected skin area"
        )
        
        st.sidebar.markdown("### 📍 Step 2: Enter Your Location")
        user_city = st.sidebar.text_input(
            "City/Location", 
            value="Cairo",
            help="Enter your city to find nearby dermatologists"
        )
        
        # Advanced options
        with st.sidebar.expander("⚙️ Advanced Options"):
            show_confidence_details = st.checkbox("Show detailed confidence scores", value=True)
            show_all_predictions = st.checkbox("Show all class probabilities", value=False)
        
        #
