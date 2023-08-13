import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="`np\\.object` was a deprecated alias for the builtin `object`")
from json.decoder import JSONDecodeError
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


MODEL = tf.keras.models.load_model("./ninetynine.h5")
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']



st.title("POTATO LEAF DISEASE PREDICTION")

uploaded_image = st.file_uploader("Upload Potato Leaf Image",type=["jpg","png","jpeg"])
if uploaded_image is not None:
    bytes_data = uploaded_image.getvalue()
    image = np.array(Image.open(BytesIO(bytes_data)).resize((224,224)))
    img_batch = np.expand_dims(image,0)
    try:
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        up_image = Image.open(uploaded_image)
        st.image(up_image, width = 300,use_column_width=True)
        st.success("Leaf is in {} condition {} and Prediction Confidence = {}%".format(predicted_class,"\n",round(confidence*100,2)))
    except Exception as e:
        st.title("Invalid Image")

teja = Image.open("pics/teja.jpg")

# Create a card with centered content
col1, col2, col3 = st.columns([3, 4, 3])
with col2:
    st.image(teja, width=200)

st.markdown("<h3 style='text-align: center;'>Tejeswara Murthy Palwadi</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px;'>Author</p>", unsafe_allow_html=True)
