import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = AutoTokenizer.from_pretrained('./sentiment_model')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #EE4B2B	;  /* Bright Red */
        font-size: 40px;
        font-weight: bold;  /* Make the title bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Title of the app
st.markdown('<p class="title">Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)

# Centering the content
with st.container():
    st.subheader("Enter Text for Sentiment Analysis")

    # Text area for user input
    input_text = st.text_area("Input Text (one sentence per line)", height=200)

    if st.button("Analyze"):
        if input_text:
            # Prepare the input data
            test_sentences = input_text.splitlines()
            inputs = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")

            # Make predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                confidences = torch.softmax(outputs.logits, dim=-1)

            # Display results
            st.subheader("Analysis Results:")
            for sentence, label, confidence in zip(test_sentences, predictions, confidences):
                sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
                confidence_score = confidence[label].item() * 100
                st.write(f"**Sentence:** {sentence}")
                st.write(f"**Predicted Sentiment:** {sentiment} ({confidence_score:.2f}%)")
                st.write("")  # For spacing
        else:
            st.error("Please enter at least one sentence for analysis.")
