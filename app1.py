import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import time
import base64

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for dark theme
def add_dark_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #0f172a !important;
        }}
        
        .css-1n76uvr, .css-18e3th9, .css-1d391kg, .css-12oz5g7 {{
            background-color: #1e293b !important;
            border-radius: 10px !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            margin-bottom: 20px !important;
            color: #e2e8f0 !important;
        }}
        
        h1, h2, h3 {{
            color: #38bdf8 !important;
            font-weight: 600 !important;
        }}
        
        p, li {{
            color: #e2e8f0 !important;
        }}
        
        .stButton>button {{
            background-color: #38bdf8 !important;
            color: #0f172a !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: bold !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }}
        
        .stButton>button:hover {{
            background-color: #0ea5e9 !important;
            transform: translateY(-2px) !important;
        }}
        
        .stProgress > div > div {{
            background-color: #38bdf8 !important;
        }}
        
        .stExpander {{
            background-color: #1e293b !important;
            border-radius: 10px !important;
        }}
        
        .glass-card {{
            background-color: #1e293b !important;
            border-radius: 10px !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            margin-bottom: 20px !important;
            color: #e2e8f0 !important;
        }}
        
        .result-card {{
            background-color: #1e293b !important;
            border-radius: 10px !important;
            padding: 20px !important;
            box-shadow: 0 4px 15px rgba(56, 189, 248, 0.1) !important;
            border: 1px solid #38bdf8 !important;
            margin-top: 20px !important;
            text-align: center !important;
        }}
        
        .stAlert {{
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
        }}
        
        .stFileUploader {{
            background-color: #1e293b !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }}
        
        .uploadedFile {{
            background-color: #1e293b !important;
        }}
        
        .css-1adrfps {{
            background-color: #1e293b !important;
        }}
        
        .css-pkbazv {{
            color: #e2e8f0 !important;
        }}
        
        .css-8ojfln {{
            color: #e2e8f0 !important;
        }}
        
        .stTextInput > div > div > input {{
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
        }}
        
        .stSelectbox > div > div > select {{
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
        }}
        
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0f172a;
            color: #94a3b8;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            border-top: 1px solid #1e293b !important;
        }}
        
        /* Spinner customization */
        .stSpinner > div > div > div {{
            border-top-color: #38bdf8 !important;
        }}
        
        /* Success message */
        .element-container div[data-testid="stDecoration"] {{
            background-color: rgba(5, 150, 105, 0.2) !important;
            border: 1px solid rgba(5, 150, 105, 0.3) !important;
            color: #d1fae5 !important;
        }}
        
        /* Warning message */
        .element-container div[data-baseweb="notification"] {{
            background-color: rgba(245, 158, 11, 0.2) !important;
            border: 1px solid rgba(245, 158, 11, 0.3) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_dark_theme()

# Helper functions
def preprocess_image(image, image_size):
    img = np.array(image)
    img = cv2.resize(img, (image_size, image_size))
    return img

def predict_image(model, image, class_names):
    preprocessed_img = preprocess_image(image, model.input_shape[1])
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)
    confidence = float(predictions[0][predicted_class_index[0]] * 100)
    predicted_class = class_names[predicted_class_index[0]]
    return predicted_class, confidence

def loadmodel():
    model_path = './brainTumor.keras'
    model = load_model(model_path, compile=False)
    return model

# App header
st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🧠 Neural Scan</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #94a3b8;'>Advanced Brain Tumor Detection System</h3>", unsafe_allow_html=True)

# Create three columns for the top section
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("""
    <div class="glass-card">
        <h3 style='text-align: center; color: #38bdf8;'>About</h3>
        <p>This AI-powered application uses deep learning to detect different types of brain tumors from MRI scans with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card" style='text-align: center;'>
        <h2>Brain Tumor Analysis</h2>
        <p>Upload an MRI scan to get instant AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
        <h3 style='text-align: center; color: #38bdf8;'>Tumor Categories</h3>
        <ul>
            <li>Glioma Tumor</li>
            <li>Meningioma Tumor</li>
            <li>Pituitary Tumor</li>
            <li>No Tumor (Healthy)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content area
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("""
    <div class="glass-card">
        <h3 style='text-align: center; color: #38bdf8;'>Upload MRI Scan</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "photo" not in st.session_state:
        st.session_state["photo"] = "Not Done"
    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None
    if "confidence" not in st.session_state:
        st.session_state["confidence"] = None

    def change_photo_state():
        st.session_state["photo"] = "Done"
     
    uploaded_pic = st.file_uploader("Choose an MRI scan image...", on_change=change_photo_state, type=["jpg","png","jpeg"])
    
    if st.session_state["photo"] == "Done":
        with st.spinner("Processing image..."):
            progress_bar = st.progress(0)
           
            for perc_completed in range(100):
                time.sleep(0.01)
                progress_bar.progress(perc_completed + 1)
                
            st.success("Image successfully uploaded!")
            
            if uploaded_pic is not None:
                test_image = Image.open(uploaded_pic)
                resized_image = test_image.resize((256, 256))
                st.markdown("""
                <div style="padding: 10px; background-color: #1e293b; border-radius: 10px; text-align: center;">
                """, unsafe_allow_html=True)
                st.image(resized_image, caption='Uploaded MRI Scan', use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #1e293b; border-radius: 10px;'>
            <p style='color: #94a3b8;'>Please upload a brain MRI image for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='color: #38bdf8; margin-top: 20px;'>Example MRI Scan:</p>", unsafe_allow_html=True)
        st.markdown("""
        <div style="padding: 10px; background-color: #1e293b; border-radius: 10px; text-align: center;">
        """, unsafe_allow_html=True)
        test_image = Image.open("./Example.jpg")
        resized_image = test_image.resize((256, 256))
        st.image(resized_image, caption='Example MRI Scan', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div class="glass-card">
        <h3 style='text-align: center; color: #38bdf8;'>Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model in a container to show loading status
    with st.spinner("Loading AI Model..."):
        try:
            model = loadmodel()
            class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            st.success("Neural network loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    analyze_button = st.button("🔍 ANALYZE SCAN", key="analyze")
    
    if analyze_button:
        if "photo" in st.session_state and st.session_state["photo"] == "Done" and 'test_image' in locals():
            with st.spinner("Analyzing brain MRI scan..."):
                # Add a simulated delay with a progress bar for better UX
                progress_text = "Running neural network analysis..."
                analysis_bar = st.progress(0)
                for i in range(100):
                    # Simulate computation time
                    time.sleep(0.02)
                    analysis_bar.progress(i + 1)
                
                predicted_class, confidence = predict_image(model, test_image, class_names)
                st.session_state["prediction"] = predicted_class
                st.session_state["confidence"] = confidence
        else:
            st.warning("Please upload an MRI scan image first")
    
    # Display results
    if "prediction" in st.session_state and st.session_state["prediction"] is not None:
        result_color = "#0ea5e9" if "no tumor" in st.session_state["prediction"].lower() else "#f43f5e"
        
        st.markdown(f"""
        <div class="result-card">
            <h2 style='color: {result_color};'>Analysis Result</h2>
            <h3 style='color: #e2e8f0;'>Diagnosis: {st.session_state["prediction"]}</h3>
            <div style="background-color: #1e293b; border-radius: 10px; padding: 10px; margin: 10px 0;">
                <p>Confidence: {st.session_state["confidence"]:.2f}%</p>
                <div style="background-color: #1e293b; border-radius: 5px; height: 10px; width: 100%; margin-top: 5px;">
                    <div style="background-color: {result_color}; width: {st.session_state["confidence"]}%; height: 10px; border-radius: 5px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional information based on result
        if "tumor" in st.session_state["prediction"].lower() and "no" not in st.session_state["prediction"].lower():
            st.markdown("""
            <div class="glass-card" style="margin-top: 20px;">
                <h4 style="color: #f43f5e;">Recommended Next Steps</h4>
                <p>This AI analysis indicates the possible presence of a brain tumor. Please consult with a healthcare professional immediately for proper medical evaluation.</p>
                <ul>
                    <li>Consult with a neurologist or neurosurgeon</li>
                    <li>Additional diagnostic tests may be required</li>
                    <li>Early treatment planning is recommended</li>
                </ul>
                <p style="font-style: italic; font-size: 0.8em; color: #94a3b8;">Note: This is an AI-assisted analysis and should not replace professional medical diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="margin-top: 20px;">
                <h4 style="color: #0ea5e9;">Result Information</h4>
                <p>The analysis suggests no detectable brain tumor in the provided MRI scan.</p>
                <ul>
                    <li>Regular follow-ups are still recommended</li>
                    <li>Maintain routine neurological check-ups</li>
                    <li>Report any new symptoms to your healthcare provider</li>
                </ul>
                <p style="font-style: italic; font-size: 0.8em; color: #94a3b8;">Note: This is an AI-assisted analysis and should not replace professional medical diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)

# Bottom section with bento grid
st.markdown("<br>", unsafe_allow_html=True)
info_col1, info_col2, info_col3 = st.columns([1, 1, 1])

with info_col1:
    with st.expander("How Neural Scan Works"):
        st.markdown("""
        This application uses a convolutional neural network (CNN) trained on thousands of MRI brain scans to detect and classify various types of brain tumors:
        
        1. **Upload** your MRI scan
        2. Our **deep learning model** processes the image through multiple neural layers
        3. Get instant **diagnostic suggestions** with confidence score
        
        The neural network has been trained to recognize patterns associated with different tumor types, enabling high-accuracy predictions even from a single scan image.
        """)

with info_col2:
    with st.expander("About Brain Tumors"):
        st.markdown("""
        **Types of Brain Tumors:**
        
        - **Glioma**: Occurs in the brain and spinal cord, originates in glial cells
        - **Meningioma**: Forms on membranes covering the brain and spinal cord, usually benign
        - **Pituitary**: Develops in the pituitary gland at the base of the brain
        
        Early detection through MRI scanning is crucial for effective treatment planning and improved outcomes.
        """)

with info_col3:
    with st.expander("Developer Information"):
        st.markdown("""
        **Contact the Developer:**
        - Email: 
        
        **GitHub Repository:**
        - [Brain Tumor Detection Project](https://github.com/2512ankita/brain-tumor-detction/tree/main)
        
        **Technical Stack:**
        - TensorFlow/Keras for deep learning
        - Streamlit for web interface
        - OpenCV for image processing
        
        **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.
        """)

# Footer
st.markdown("""
<div class="footer">
    © 2025 Neural Scan | Brain Tumor Detection System | Developed with ❤️ | Powered by TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)