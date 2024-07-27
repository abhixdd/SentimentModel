import re
import joblib

# Load the saved model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Preprocess the input text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    return re.sub(r'[^\w\s]', '', text)  # Remove punctuation

# Prediction function
def predict_sentiment(text):
    # Preprocess the input text
    text = preprocess_text(text)
    
    # Convert text to numerical data
    text_vectorized = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(text_vectorized)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return sentiment

# Example usage
if __name__ == "__main__":
    example_text = "I am so  happy with the results!"
    print(f"Sentiment: {predict_sentiment(example_text)}")
