import streamlit as st
import re
import numpy as np
import joblib

def preprocess_text(text):
    return re.sub('[-@~%^&*+#$/\.?!<>;:,\'\"\\(){}]', ' ', text).lower()

def load_model(model_path):
    return joblib.load(model_path)

def predict_category(model, vectorizer, text):
    text_processed = preprocess_text(text)
    text_vectorized = vectorizer.transform([text_processed])
    prediction = model.predict(text_vectorized)
    return prediction[0]

def main():
    st.title('Bima Aji Sakti Craigslist Post Classifier')
    st.write('Enter your post heading to predict the category:')

    model_path = 'trained_model_linear_svc.pkl'
    clf, vectorizer = load_model(model_path)

    text_input = st.text_input('Enter post heading:', '')

    if st.button('Predict'):
        if text_input:
            category = predict_category(clf, vectorizer, text_input)
            st.write(f'Predicted category: {category}')
        else:
            st.write('Please enter a post heading.')

if __name__ == '__main__':
    main()
