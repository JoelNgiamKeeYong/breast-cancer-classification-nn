import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="centered"
)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = keras.models.load_model('breast_cancer_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Feature names for input
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Function to generate realistic dummy data
def generate_dummy_data():
    # Define base ranges for different measurement types
    base_ranges = {
        'radius': (6, 28),
        'texture': (9, 39),
        'perimeter': (43, 188),
        'area': (143, 2501),
        'smoothness': (0.053, 0.163),
        'compactness': (0.019, 0.345),
        'concavity': (0, 0.427),
        'concave points': (0, 0.201),
        'symmetry': (0.106, 0.304),
        'fractal dimension': (0.05, 0.097),
    }
    
    dummy_data = {}
    
    for feature in feature_names:
        # Split feature name into parts
        parts = feature.split()
        
        # Get the base measurement type
        if 'concave points' in feature:
            base_feature = 'concave points'
        elif 'fractal dimension' in feature:
            base_feature = 'fractal dimension'
        else:
            # For other features, the base feature is the last word that isn't 'error'
            # Skip 'mean' or 'worst' prefix if present
            base_feature = parts[-1] if parts[-1] != 'error' else parts[-2]
        
        # Get the range for this base feature
        min_val, max_val = base_ranges[base_feature]
        
        # Generate appropriate value based on feature type
        if 'error' in feature:
            # Errors are typically smaller than the measurements
            dummy_data[feature] = np.random.uniform(0, (max_val - min_val) * 0.1)
        elif 'worst' in feature:
            # Worst values are typically higher than means
            mean_val = np.random.uniform(min_val, max_val)
            dummy_data[feature] = mean_val * np.random.uniform(1, 1.5)
        else:
            # Mean values
            dummy_data[feature] = np.random.uniform(min_val, max_val)
    
    return dummy_data

def main():
    # [Previous CSS styles remain the same]
    st.markdown("""
        <style>
        .main {
            padding: 0rem 2rem;
        }
        .stAlert {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        h1 {
            color: #ff4b4b;
        }
        h2 {
            color: #ff4b4b;
            opacity: 0.8;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🔬 Breast Cancer Classification System")
    st.markdown("An AI-powered tool utilising neural networks for breast cancer tumor classification")
    st.markdown("[View on GitHub](https://github.com/JoelNgiamKeeYong/breast-cancer-classification-nn)")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Info"])

    if page == "Home":
        show_home_page()
    elif page == "Prediction":
        show_prediction_page()
    elif page == "Model Info":
        show_model_info()

def show_prediction_page():
    st.header("Tumor Classification Prediction")
    
    # Add Generate Dummy Data button above tabs with custom styling
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            margin: 0 auto;
            display: block;
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Generate dummy data if button is clicked
    if st.button("🎲 Generate Sample Data"):
        st.session_state.dummy_data = generate_dummy_data()
        st.success("✅ Sample data generated successfully!")
    
    # Create tabs for different input methods
    tab1 = st.tabs(["📝 Manual Input"])
    
    with tab1:
        st.subheader("Enter Cell Nucleus Measurements")
        
        # Create 3 columns for input layout
        col1, col2, col3 = st.columns(3)
        
        # Initialize input dictionary
        input_dict = {}
        
        # Create input fields for each feature
        for i, feature in enumerate(feature_names):
            with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                default_value = st.session_state.dummy_data[feature] if hasattr(st.session_state, 'dummy_data') else 0.0
                input_dict[feature] = st.number_input(
                    f"{feature}",
                    value=float(default_value),
                    format="%.6f"
                )

        if st.button("Predict", key="predict_single"):
            input_data = np.array(list(input_dict.values())).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            prediction_label = int(np.argmax(prediction))
            prediction_prob = prediction[0][prediction_label] * 100

            st.markdown("### Results")
            if prediction_label == 0:
                st.error(f"🚨 Prediction: Malignant (Confidence: {prediction_prob:.2f}%)")
            else:
                st.success(f"✅ Prediction: Benign (Confidence: {prediction_prob:.2f}%)")

            st.subheader("Feature Values Visualization")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_names,
                y=input_data[0],
                name="Feature Values"
            ))
            fig.update_layout(
                title="Input Feature Values",
                xaxis_tickangle=-45,
                height=500
            )
            st.plotly_chart(fig)
    

def show_home_page():
    st.header("Welcome to the Breast Cancer Classification System")
    
    # Introduction
    st.markdown("""
    This application uses machine learning to help classify breast cancer tumors as either benign or malignant
    based on various cell nucleus characteristics. The model has been trained on the Wisconsin Breast Cancer dataset.
    
    ### Key Features:
    - Real-time prediction of tumor classification
    - Interactive visualization of input features
    - Detailed model performance metrics
    - Educational resources about breast cancer diagnosis
    """)

    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", "569")
    with col2:
        st.metric("Features", "30")
    with col3:
        st.metric("Model Accuracy", "96.5%")

    # Sample Feature Visualization
    st.subheader("Sample Feature Distributions")
    df = pd.DataFrame(scaler.inverse_transform(np.random.normal(0, 1, (100, 30))), columns=feature_names)
    
    selected_feature = st.selectbox("Select a feature to visualize:", feature_names)
    fig = px.histogram(df, x=selected_feature, nbins=30)
    st.plotly_chart(fig)

def show_prediction_page():
    st.header("Tumor Classification Prediction")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["🎲 1. Generate Dummy Data", "📝 2. Predict"])
    
    with tab1:
        st.subheader("Generate Sample Data")
        if st.button("Generate Random Sample", key="generate_dummy"):
            dummy_data = generate_dummy_data()
            st.session_state.dummy_data = dummy_data
            st.success("✅ Sample data generated successfully!")
            
            # Display the generated data in a table format
            st.subheader("Generated Sample Data")
            df_dummy = pd.DataFrame([dummy_data])
            st.dataframe(df_dummy)
            
            # Create visualization of the generated data
            st.subheader("Data Visualization")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(dummy_data.keys()),
                y=list(dummy_data.values()),
                name="Generated Values"
            ))
            fig.update_layout(
                title="Generated Feature Values",
                xaxis_tickangle=-45,
                height=500
            )
            st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Enter Cell Nucleus Measurements")
        
        # Create 3 columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Initialize input dictionary
        input_dict = {}
        
        # Create input fields for each feature
        for i, feature in enumerate(feature_names):
            with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                default_value = st.session_state.dummy_data[feature] if hasattr(st.session_state, 'dummy_data') else 0.0
                input_dict[feature] = st.number_input(
                    f"{feature}",
                    value=float(default_value),
                    format="%.6f"
                )

        if st.button("Predict", key="predict_single"):
            # Prepare input data
            input_data = np.array(list(input_dict.values())).reshape(1, -1)
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_label = int(np.argmax(prediction))
            prediction_prob = prediction[0][prediction_label] * 100

            # Display result with custom styling
            st.markdown("### Results")
            if prediction_label == 0:
                st.error(f"🚨 Prediction: Malignant (Confidence: {prediction_prob:.2f}%)")
            else:
                st.success(f"✅ Prediction: Benign (Confidence: {prediction_prob:.2f}%)")

            # Display feature importance visualization
            st.subheader("Feature Values Visualization")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_names,
                y=input_data[0],
                name="Feature Values"
            ))
            fig.update_layout(
                title="Input Feature Values",
                xaxis_tickangle=-45,
                height=500
            )
            st.plotly_chart(fig)
    

def show_model_info():
    st.header("Model Information")
    
    # Model Architecture
    st.subheader("Model Architecture")
    st.code("""
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')
    ])
    """)
    
    # Training Parameters
    st.subheader("Training Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Optimizer", "Adam")
    with col2:
        st.metric("Loss Function", "Sparse Categorical Crossentropy")
    with col3:
        st.metric("Epochs", "10")

    # Performance Metrics
    st.subheader("Model Performance")
    
    # Create sample metrics (replace with actual values from your model)
    metrics = {
        'Training Accuracy': 0.965,
        'Validation Accuracy': 0.955,
        'Test Accuracy': 0.960
    }
    
    # Display metrics
    cols = st.columns(len(metrics))
    for col, (metric, value) in zip(cols, metrics.items()):
        col.metric(metric, f"{value:.1%}")

if __name__ == "__main__":
    main()