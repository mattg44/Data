import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import streamlit as st

st.title("Tomato diseases and disorders")
st.subheader("Upload an image of a tomato leaf (clear background required)")

class_names = ["bacterial spot", "early blight", "healthy tomato, no disease", "late blight", "leaf mold",
               "septoria leaf spot", "red spider mites", "target spot", "mosaic virus", "yellow leaf curf virus"]


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model/model_nolf")
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])


def resize(image_data):
    size = (64, 64)
    image_to_resize = np.array(ImageOps.fit(image_data, size=size, method=Image.ANTIALIAS))
    image_to_resize = image_to_resize.reshape(-1, 64, 64, 3) / 255
    # img_resize = (cv2.resize(image, dsize=(34, 34), interpolation=cv2.INTER_CUBIC)) / 255.
    return image_to_resize

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    resized_image = resize(image)
    predictions = model.predict(resized_image)
    score = np.argmax(predictions)
    st.subheader(f"Deseases: {class_names[score]} with a confidence of {np.max(predictions) * 100:.2f} %")
