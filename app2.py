import streamlit as st
import pickle
import torch

# Load the saved model and tokenizer from pickle files
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit app
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")
st.title("**Sentiment Analysis Dashboard**", anchor="title")

# Create a text input for user input
st.markdown("### Enter your text below for sentiment analysis:")
input_text = st.text_area("Input Text", height=150)

# Define a function for making predictions
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Get prediction probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = torch.max(probabilities).item() * 100  # Convert to percentage
        
    return "POSITIVE" if predictions.item() == 1 else "NEGATIVE", confidence

# Create a button to submit the input text
if st.button("Analyze Sentiment"):
    if input_text:
        sentiment, confidence = predict_sentiment(input_text)
        st.markdown(f"### Predicted Sentiment: **{sentiment}**")
        st.markdown(f"### Confidence: **{confidence:.2f}%**")
    else:
        st.markdown("Please enter some text for analysis.")

# Add some styling
st.markdown(
    """
    <style>
    .css-18ni7ap.e1tzin5v0 {
        font-size: 20px;
        color: #1E90FF; /* Change color to a professional blue */
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
