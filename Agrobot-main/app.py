from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from keras.models import load_model
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load necessary files
model = load_model('model.keras')
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('data.json') as json_data:
    intents = json.load(json_data)

# Define the home route to render the chatbot interface
@app.route('/')
def home():
    return render_template('index.html')

# Define the chat route to handle user messages
@app.route('/chat', methods=['POST'])
def chat():
    msg = request.form['msg']
    if msg:
        # Process the user message and generate a response
        ints = predict_class(msg, model)
        res = get_response(ints, intents)
        return jsonify({'response': res})

# Function to predict the class
def predict_class(sentence, model):
    # Tokenize and lemmatize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # Create a bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    # Convert to NumPy array and predict
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get the response
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

if __name__ == '__main__':
    app.run(debug=True)
