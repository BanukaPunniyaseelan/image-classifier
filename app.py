import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

@st.cache_resource
def load_model():
    return keras.models.load_model("image_classifier.keras")

model = load_model()

st.title("CIFAR-10 Image Classifier")
st.write("Upload a photo and the model will guess which of 10 categories it belongs to.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    resized = img.resize((32, 32))
    arr = np.array(resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr)[0]
    top3 = np.argsort(probs)[::-1][:3]

    st.subheader("Predictions")
    for i in top3:
        st.write(f"**{class_names[i]}**: {probs[i]:.1%}")