import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))
sys.path.append(str('/app/fake-detector/api'))
sys.path.append(str('/api'))
sys.path.append(str('/app/api'))

from model_inference import explainable_layers


uploaded_file = st.file_uploader("Upload a picture of a person", type="jpg")
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    #st.write(f"{prediction}")
    st.write("Classifying...")
    prediction = fake_detector(image)
    prediction_pct = prediction * 100

    label = np.argmax(prediction)
    formatted_pct = round(float(prediction_pct[0][label]),3)
    if label == 0:
        st.write(f"AI generated picture: model certainty of {formatted_pct}%")
    elif label ==1:
        st.write(f"Authentic picture: model certainty of {formatted_pct}%")

    st.pyplot(explainable_layers(image))
    
