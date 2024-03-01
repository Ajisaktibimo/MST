import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

def preprocess_text(text):
    return re.sub('[-@~%^&*+#$/\.?!<>;:,\'\"\\(){}]', ' ', text).lower()

def main():
    training_file = 'training.json'
    data, categories = load_data(training_file)

    clf, vectorizer = train_model(data, categories)

    model_file = 'trained_model_linear_svc.pkl'
    save_model(clf, vectorizer, model_file)

def load_data(file_path):
    data = []
    categories = []
    with open(file_path, 'r') as file:
        next(file)  
        for line in file:
            post = json.loads(line)
            data.append(preprocess_text(post['heading']))
            categories.append(post['category'])
    return data, categories

def train_model(data, categories):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=0, max_df=.2, ngram_range=(1, 2), max_features=9000, preprocessor=preprocess_text)
    X_train = vectorizer.fit_transform(data)

    clf = LinearSVC()
    clf.fit(X_train, categories)
    return clf, vectorizer

def save_model(clf, vectorizer, file_path):
   
    joblib.dump((clf, vectorizer), file_path)
    print('Model saved successfully.')

if __name__ == '__main__':
    main()
