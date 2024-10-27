# Sentiment Analysis with Transformers 🧠✨

Welcome to the **Sentiment Analysis** project! This application uses state-of-the-art transformer models to classify text sentiment as either positive or negative. The model is fine-tuned to deliver high accuracy in predicting sentiments based on input text.

## Table of Contents 📚

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Example Sentences](#example-sentences)

  
## Features 🌟

- User-friendly web interface for easy input and output.
- Real-time sentiment prediction with confidence scores.
- Supports multiple input sentences at once.
- Built with Streamlit for interactive and responsive UI.

## Technologies Used 🛠️

- **Python**: Programming language used for the project.
- **Transformers**: Hugging Face's library for state-of-the-art NLP.
- **Streamlit**: Framework for building web applications easily.
- **PyTorch**: Deep learning framework used for model implementation.

## Example Sentences 📝

Here are some example sentences you can test with:

```python
test_sentences = [
    "I absolutely love this product! It's amazing.",  # ❤️
    "This is the worst experience I've ever had.",      # 😡
    "The service was okay, nothing special.",          # 😐
    "I'm very satisfied with my purchase."              # 😊
]

# Usage in your application:
for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    # Here you would call your sentiment analysis function
    # For example:
    # predicted_label, confidence = analyze_sentiment(sentence)
    # print(f"Predicted Sentiment: {predicted_label} with confidence {confidence:.2f}%\n")
```


## Model Files 🗂️

The following files are included in the `sentiment_model` folder:

- `tokenizer.pkl`: The tokenizer used for preprocessing the text data.
- `sentiment_model.pkl`: The trained sentiment analysis model.
- `model.safetensors`: Model weights stored in the SafeTensors format.

**Important:** The dataset files used in this project are large. GitHub does not allow file sizes greater than 25MB. If your files exceed this limit, consider using alternative hosting solutions such as [Git LFS](https://git-lfs.github.com/) or [Google Drive](https://drive.google.com/).
