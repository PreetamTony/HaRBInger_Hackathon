import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from gtts import gTTS
import base64
import firebase_admin
from firebase_admin import credentials, firestore, auth

# Initialize Firebase Admin SDK (ensure this is done only once)
if not firebase_admin._apps:
    cred = credentials.Certificate("C:/Users/preet/OneDrive/Desktop/Harbinger Hackathon/Model2/money-recognition-3ab7b-firebase-adminsdk-woubn-040bf60023.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the model
model = load_model("C:/Users/preet/OneDrive/Desktop/Harbinger Hackathon/Model2/keras_model.h5", compile=False)

# Load the labels
class_names = open("C:/Users/preet/OneDrive/Desktop/Harbinger Hackathon/Model2/labels.txt", "r").readlines()

# Function to predict image class
def model_predict(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:].strip(), confidence_score

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text, lang=lang)
    tts.save("uploads/voice.mp3")
    return "uploads/voice.mp3"

# Function to generate download link for audio file
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:audio/mp3;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# Function to save transaction to Firestore
def save_transaction(user_email, class_name, confidence_score):
    # Convert confidence_score to a basic Python type, like float
    confidence_score = float(confidence_score)

    transaction = {
        'user_email': user_email,
        'class_name': class_name,
        'confidence_score': confidence_score,
        'timestamp': firestore.SERVER_TIMESTAMP
    }
    db.collection('transactions').add(transaction)

# Function to fetch and display saved transactions
def display_transactions():
    st.subheader("Previous Transactions")
    transactions_ref = db.collection('transactions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10)
    docs = transactions_ref.get()
    for doc in docs:
        transaction_data = doc.to_dict()
        st.write(f"User: {transaction_data['user_email']}")
        st.write(f"Class Name: {transaction_data['class_name']}")
        st.write(f"Confidence Score: {transaction_data['confidence_score']:.2f}")
        st.write(f"Timestamp: {transaction_data['timestamp']}")

# Streamlit app
st.set_page_config(page_title="Money Classifier for Blind Users", layout="wide")

# Header
st.title("Money Classifier for Blind Users")

# Logo
st.image("https://i.pinimg.com/236x/23/63/a9/2363a981f0950d9b6b0d4c99b60f01c0.jpg", use_column_width=True)

# Login and Signup Forms
st.subheader("Login")
user_email = st.text_input("Enter your email:", key="login_email")
user_password = st.text_input("Enter your password:", type='password', key="login_password")

if st.button("Login"):
    try:
        user = auth.get_user_by_email(user_email)
        st.success(f"Welcome {user.email}")
    except Exception as e:
        st.error("Authentication failed. Please check your credentials.")
        st.error(f"Error details: {str(e)}")

st.subheader("Sign Up")
new_email = st.text_input("Enter new email:", key="signup_email")
new_password = st.text_input("Enter new password:", type='password', key="signup_password")

if st.button("Sign Up"):
    try:
        user = auth.create_user(email=new_email, password=new_password)
        st.success(f"User {user.email} successfully created!")
    except Exception as e:
        st.error(f"Failed to create user. Error: {str(e)}")

# Upload and Classify Images Section
st.subheader("Upload and Classify Images")
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        class_name, confidence_score = model_predict(file_path)

        st.markdown(f"**Class:** {class_name}")
        st.markdown(f"**Confidence Score:** {confidence_score:.2f}")

        speech_file = text_to_speech(f"The predicted class is {class_name} with a confidence score of {confidence_score:.2f}")
        st.audio(speech_file, format='audio/mp3')
        st.markdown(get_binary_file_downloader_html(speech_file, 'Voice Output'), unsafe_allow_html=True)

        if user_email:
            save_transaction(user_email, class_name, confidence_score)

# Display previous transactions
display_transactions()
