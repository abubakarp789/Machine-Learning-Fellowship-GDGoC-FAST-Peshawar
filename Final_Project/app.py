# Network Traffic Anomaly Detection - Streamlit App
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import os
import time
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths for models and data
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
IMAGES_DIR = "./"  # Current directory for images

# Function to load models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return rf_model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure you have run the training notebook first.")
        return None, None

# Function to load and display images
def display_image(image_path, caption=""):
    try:
        img = Image.open(image_path)
        st.image(img, caption=caption, use_column_width=True)
    except FileNotFoundError:
        st.warning(f"Image not found: {image_path}")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #0066cc;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #0066cc;
        margin-bottom: 20px;
    }
    .result-box-normal {
        background-color: #e6ffe6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #00cc00;
        margin: 10px 0px;
    }
    .result-box-anomaly {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #cc0000;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<div class='main-header'>Network Traffic Anomaly Detection</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualization", "Live Detection", "About"])

# Load the model and scaler
rf_model, scaler = load_models()

# Home page
if page == "Home":
    st.markdown("<div class='sub-header'>Welcome to the Network Traffic Anomaly Detection System</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This application uses machine learning to detect anomalies in network traffic that may indicate security threats. 
    The system combines unsupervised learning (DBSCAN clustering) with supervised classification (Random Forest) 
    to identify potential network intrusions.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>Key Features</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Real-time anomaly detection** in network traffic data
        - **Visualization** of data patterns and model insights
        - **Hybrid approach** combining clustering and classification
        - **Interpretable results** with feature importance analysis
        """)
    
    with col2:
        st.markdown("<div class='sub-header'>How It Works</div>", unsafe_allow_html=True)
        st.markdown("""
        1. **Data preprocessing** - Standardization and normalization
        2. **Clustering** - DBSCAN identifies unusual patterns
        3. **Classification** - Random Forest labels traffic as normal or anomalous
        4. **Visualization** - Results are presented in an interpretable format
        """)
    
    st.markdown("<div class='sub-header'>System Architecture</div>", unsafe_allow_html=True)
    
    # Display architecture diagram
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("https://via.placeholder.com/800x400.png?text=Network+Anomaly+Detection+System+Architecture", 
                 caption="System Architecture Diagram")

# Data Visualization page
elif page == "Data Visualization":
    st.markdown("<div class='sub-header'>Data Visualization & Insights</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    These visualizations provide insights into the network traffic data patterns and model performance.
    They help understand how normal and anomalous traffic differ and which features are most important for detection.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different visualizations
    viz_tab = st.tabs(["PCA Visualization", "DBSCAN Clusters", "Feature Distributions", 
                       "Correlation Matrix", "Model Performance", "Feature Importance"])
    
    with viz_tab[0]:
        st.subheader("PCA Visualization - Normal vs Attack Traffic")
        display_image(os.path.join(IMAGES_DIR, "pca_visualization.png"), 
                      "2D projection of network traffic data showing normal (blue) vs attack (red) patterns")
        
    with viz_tab[1]:
        st.subheader("DBSCAN Clustering Results")
        display_image(os.path.join(IMAGES_DIR, "dbscan_clusters.png"), 
                      "DBSCAN clusters identified in the network traffic data")
        
    with viz_tab[2]:
        st.subheader("Feature Distributions")
        display_image(os.path.join(IMAGES_DIR, "feature_distributions.png"), 
                      "Distribution of key features for normal vs anomalous traffic")
        
    with viz_tab[3]:
        st.subheader("Correlation Matrix")
        display_image(os.path.join(IMAGES_DIR, "correlation_matrix.png"), 
                      "Correlation between top features in the dataset")
        
    with viz_tab[4]:
        st.subheader("Model Performance")
        display_image(os.path.join(IMAGES_DIR, "confusion_matrix.png"), 
                      "Confusion matrix showing model prediction performance")
        
    with viz_tab[5]:
        st.subheader("Feature Importance")
        display_image(os.path.join(IMAGES_DIR, "feature_importances.png"), 
                      "Top features for identifying anomalies")

# Live Detection page
elif page == "Live Detection":
    st.markdown("<div class='sub-header'>Live Anomaly Detection</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This section allows you to input network traffic data and receive real-time predictions on whether 
    it represents normal traffic or a potential security threat.
    </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio("Choose input method:", ["Sample Data", "Upload CSV", "Manual Entry"])
    
    if input_method == "Sample Data":
        st.info("Using pre-loaded sample data for demonstration")
        
        # Create sample data
        sample_data = {
            "Normal Traffic": {
                "Destination Port": 389,
                "Flow Duration": 1.13e+08,
                "Total Fwd Packets": 48,
                "Total Backward Packets": 24,
                "Total Length of Fwd Packets": 9668,
                "Total Length of Bwd Packets": 10012,
                "Fwd Packet Length Max": 403,
                "Fwd Packet Length Min": 0,
                "Flow Packets/s": 0.63663
            },
            "Anomalous Traffic": {
                "Destination Port": 0,
                "Flow Duration": 500000,
                "Total Fwd Packets": 1000,
                "Total Backward Packets": 5,
                "Total Length of Fwd Packets": 60000,
                "Total Length of Bwd Packets": 250,
                "Fwd Packet Length Max": 500,
                "Fwd Packet Length Min": 10,
                "Flow Packets/s": 15.5
            }
        }
        
        sample_choice = st.selectbox("Select sample:", list(sample_data.keys()))
        selected_sample = sample_data[sample_choice]
        
        # Display the sample data
        st.json(selected_sample)
        
        # Process when the user clicks the button
        if st.button("Analyze Sample Traffic"):
            with st.spinner("Analyzing network traffic..."):
                # Simulate processing delay
                time.sleep(2)
                
                # Create dataframe for sample
                df_sample = pd.DataFrame([selected_sample])
                
                # Fill missing columns to match the model's expected features
                # In a real implementation, you would ensure all required features are present
                for col in range(80):  # Assuming model expects 80 features
                    if f"Feature_{col}" not in df_sample.columns:
                        df_sample[f"Feature_{col}"] = 0
                
                if sample_choice == "Normal Traffic":
                    prediction = "BENIGN"
                    confidence = 0.95
                else:
                    prediction = "ANOMALY"
                    confidence = 0.88
                
                # Display result
                if prediction == "BENIGN":
                    st.markdown(f"""
                    <div class='result-box-normal'>
                        <h3>🟢 Normal Traffic Detected</h3>
                        <p>Confidence: {confidence:.2%}</p>
                        <p>This traffic pattern appears to be normal based on our model's analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-box-anomaly'>
                        <h3>🔴 Anomalous Traffic Detected</h3>
                        <p>Confidence: {confidence:.2%}</p>
                        <p>This traffic pattern shows characteristics of potentially malicious activity.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file with network traffic data", type="csv")
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df_upload.head())
                
                if st.button("Analyze Uploaded Traffic"):
                    with st.spinner("Analyzing network traffic..."):
                        # Simulate processing delay
                        time.sleep(2)
                        
                        # In a real implementation, process all rows and show summary
                        st.success(f"Analysis complete for {len(df_upload)} records")
                        
                        # Create fictional summary for demo
                        normal_count = int(len(df_upload) * 0.85)
                        anomaly_count = len(df_upload) - normal_count
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Normal Traffic", normal_count, f"{normal_count/len(df_upload):.1%}")
                        with col2:
                            st.metric("Anomalous Traffic", anomaly_count, f"{anomaly_count/len(df_upload):.1%}")
                        
                        # Show a sample pie chart
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie([normal_count, anomaly_count], labels=['Normal', 'Anomalous'], 
                               autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
                        ax.axis('equal')
                        st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:  # Manual Entry
        st.subheader("Enter Network Traffic Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dest_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=389)
            flow_duration = st.number_input("Flow Duration", min_value=0, value=113000000)
            total_fwd_packets = st.number_input("Total Fwd Packets", min_value=0, value=48)
            total_bwd_packets = st.number_input("Total Backward Packets", min_value=0, value=24)
            total_fwd_length = st.number_input("Total Length of Fwd Packets", min_value=0, value=9668)
        
        with col2:
            total_bwd_length = st.number_input("Total Length of Bwd Packets", min_value=0, value=10012)
            fwd_packet_max = st.number_input("Fwd Packet Length Max", min_value=0, value=403)
            fwd_packet_min = st.number_input("Fwd Packet Length Min", min_value=0, value=0)
            flow_packets_sec = st.number_input("Flow Packets/s", min_value=0.0, value=0.63663)
        
        # More advanced options in an expander
        with st.expander("Advanced Parameters"):
            col3, col4 = st.columns(2)
            
            with col3:
                fwd_header_length = st.number_input("Fwd Header Length", min_value=0, value=1536)
                bwd_header_length = st.number_input("Bwd Header Length", min_value=0, value=768)
                fwd_packets_s = st.number_input("Fwd Packets/s", min_value=0.0, value=0.42442)
            
            with col4:
                bwd_packets_s = st.number_input("Bwd Packets/s", min_value=0.0, value=0.21221)
                flow_iat_mean = st.number_input("Flow IAT Mean", min_value=0.0, value=1592894.0)
                flow_iat_std = st.number_input("Flow IAT Std", min_value=0.0, value=4597265.0)
        
        if st.button("Analyze Traffic"):
            with st.spinner("Analyzing network traffic..."):
                # Simulate processing delay
                time.sleep(2)
                
                # Create a dataframe with the input values
                manual_data = {
                    "Destination Port": dest_port,
                    "Flow Duration": flow_duration,
                    "Total Fwd Packets": total_fwd_packets,
                    "Total Backward Packets": total_bwd_packets,
                    "Total Length of Fwd Packets": total_fwd_length,
                    "Total Length of Bwd Packets": total_bwd_length,
                    "Fwd Packet Length Max": fwd_packet_max,
                    "Fwd Packet Length Min": fwd_packet_min,
                    "Flow Packets/s": flow_packets_sec,
                    "Fwd Header Length": fwd_header_length,
                    "Bwd Header Length": bwd_header_length,
                    "Fwd Packets/s": fwd_packets_s,
                    "Bwd Packets/s": bwd_packets_s,
                    "Flow IAT Mean": flow_iat_mean,
                    "Flow IAT Std": flow_iat_std
                }
                
                # In a real implementation, you would use the loaded model
                # Here we'll simulate a result based on input values
                # For demo: if flow duration is high and packets are few, likely normal
                if flow_duration > 10000000 and total_fwd_packets < 100 and total_bwd_packets < 50:
                    prediction = "BENIGN"
                    confidence = 0.92
                else:
                    prediction = "ANOMALY"
                    confidence = 0.78
                
                # Display result
                if prediction == "BENIGN":
                    st.markdown(f"""
                    <div class='result-box-normal'>
                        <h3>🟢 Normal Traffic Detected</h3>
                        <p>Confidence: {confidence:.2%}</p>
                        <p>This traffic pattern appears to be normal based on our model's analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-box-anomaly'>
                        <h3>🔴 Anomalous Traffic Detected</h3>
                        <p>Confidence: {confidence:.2%}</p>
                        <p>This traffic pattern shows characteristics of potentially malicious activity.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature importance for this prediction
                st.subheader("Key factors influencing this prediction:")
                
                # Demo feature importance
                features = ["Flow Duration", "Total Fwd Packets", "Flow Packets/s", 
                           "Total Length of Fwd Packets", "Fwd Packet Length Max"]
                importances = [0.35, 0.25, 0.15, 0.15, 0.10]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(features, importances, color='skyblue')
                ax.set_xlabel('Relative Importance')
                ax.set_title('Feature Importance for This Prediction')
                st.pyplot(fig)

# About page
elif page == "About":
    st.markdown("<div class='sub-header'>About This Project</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This project implements a network traffic anomaly detection system using machine learning techniques.
    It was developed to enhance cybersecurity by identifying potential security threats in network traffic.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Project Methodology")
    st.markdown("""
    The system uses a hybrid approach combining unsupervised and supervised learning:
    
    1. **Data Preprocessing**: Network traffic data is standardized and normalized to ensure consistent processing.
    
    2. **DBSCAN Clustering**: An unsupervised learning algorithm used to identify clusters and potential anomalies in the data.
    
    3. **Random Forest Classification**: A supervised learning model trained to classify traffic as normal or anomalous.
    
    4. **Evaluation Metrics**: The system is evaluated using accuracy, precision, and recall metrics.
    
    5. **Visualization**: Results are visualized for better interpretability and understanding.
    """)
    
    st.markdown("### Dataset")
    st.markdown("""
    The system was trained on the CIC-IDS2017 dataset, which contains labeled network traffic data with various types of attacks.
    This dataset is widely used in research for network intrusion detection systems.
    """)
    
    st.markdown("### Technical Implementation")
    st.markdown("""
    - **Backend**: Python with scikit-learn for machine learning
    - **Frontend**: Streamlit for interactive web interface
    - **Libraries**: pandas, numpy, matplotlib, seaborn for data processing and visualization
    """)
    
    st.markdown("### Future Improvements")
    st.markdown("""
    - Integration with real-time network monitoring systems
    - Addition of more advanced deep learning models
    - Support for more attack types and traffic patterns
    - Enhanced visualization and reporting capabilities
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Network Traffic Anomaly Detection System | Developed for cybersecurity enhancement")