import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model with error handling
@st.cache_resource  # Cache the model for better performance
def load_keras_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        return model
    except IOError:
        st.error(f"Model file not found: {model_path}")
        return None

# Load the class names with error handling
@st.cache_data  # Cache the class names
def load_class_names(labels_path):
    try:
        with open(labels_path, "r") as file:
            class_names = file.readlines()
        return class_names
    except IOError:
        st.error(f"Labels file not found: {labels_path}")
        return None

# Preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# Predict using the loaded model
def predict(model, image, class_names):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = preprocess_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Streamlit UI
def main():
    st.set_page_config(page_title="Banknote Recognition", page_icon="💵", layout="wide")
    st.title('Banknote Recognition For Blind People')
    st.write('Upload an image of a banknote or use your camera to classify its denomination.')

    model_path = "C:/Users/preet/OneDrive/Desktop/Harbinger Hackathon/keras_Model.h5"  # Update to your model's path
    labels_path = "C:/Users/preet/OneDrive/Desktop/Harbinger Hackathon/labels.txt"    # Update to your labels' path

    model = load_keras_model(model_path)
    class_names = load_class_names(labels_path)

    if model is None or class_names is None:
        return

    st.sidebar.header("Options")
    option = st.sidebar.radio("Choose input method:", ('Upload Image', 'Use Camera'))

    if option == 'Upload Image':
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Classify Banknote'):
                class_name, confidence_score = predict(model, image, class_names)
                st.success(f'Class: {class_name}')
                st.info(f'Confidence Score: {confidence_score:.2f}')
    elif option == 'Use Camera':
        st.info("Press 'Start' to begin camera feed.")
        
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.model = model
                self.class_names = class_names

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                class_name, confidence_score = predict(self.model, img_pil, self.class_names)
                label = f"{class_name}: {confidence_score:.2f}"
                img = cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                return img

        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if __name__ == '__main__':
    main()
