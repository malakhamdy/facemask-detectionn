import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ----------- Load the trained model -----------
@st.cache_resource
def load_model():
    model_path = r"C:\Users\malak\Downloads\facce detection"
    if not os.path.exists(model_path):
        st.error(f"❌ Model folder not found at: {model_path}")
        st.stop()
    try:
        # Load SavedModel as a Keras layer
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224,224,3)),  # adjust input size
            tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        ])
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()
