# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

# Add a custom header image
header_image = Image.open('header_image.jpeg')  
st.image(header_image, use_column_width=True)

st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-family: 'Helvetica', sans-serif;
    color: #F63366;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Enter a movie review to classify it as positive or negative.</p>', unsafe_allow_html=True)

# User input
user_input = st.text_area('🎥 Your Movie Review Here:', height=200, help="Write your thoughts about the movie.")

# Add a slider to adjust text padding length
padding_length = st.slider("Adjust Review Padding Length", 100, 500, 500)

if st.button('🎯 Classify'):
    with st.spinner('Analyzing...'):
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = '😊 Positive' if prediction[0][0] > 0.5 else '😞 Negative'

        # Display the result
        st.success(f'Sentiment: {sentiment}')
        st.info(f'Prediction Confidence Score: {prediction[0][0]:.2f}')
        st.balloons()
else:
    st.write('✍️ Please enter a movie review.')

# Adding an expander for explanation of the model
with st.expander("ℹ️ About this App"):
    st.write("""
    This application uses a simple Recurrent Neural Network (RNN) model trained on the IMDB movie reviews dataset to classify reviews as positive or negative.
    The model is capable of understanding the sentiment behind the reviews you write. 
    Just type in your review and hit "Classify" to see the results!
    """)
