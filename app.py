from flask import Flask, request, render_template
import numpy as np
import pickle as pk
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import tensorflow.keras.preprocessing.text as tf_keras_text
sys.modules['keras.preprocessing.text'] = tf_keras_text
sys.modules['keras.src.preprocessing.text'] = tf_keras_text  

try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pk.load(handle)
    print("Tokenizer loaded successfully!")

except Exception as e:
    print("Could not load old tokenizer:", e)
    print("Recreating a new tokenizer as fallback.")
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

app = Flask(__name__)

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = request.form.get("text")
        return f"You entered: {data}"
    return render_template("home.html")

@app.route("/predicted_data", methods=["POST"])
def predict():
    model = load_model("GRU_model.h5")
    max_sequence = model.input_shape[1] + 1

    text = request.form.get("text")

    if not text:
        return "No input text provided!"

    next_word = predict_next_word(model=model, tokenizer=tokenizer, text=text, max_sequence_len=max_sequence)

    if next_word:
        return f"Next predicted word: <b>{next_word}</b>"
    else:
        return "Couldn't predict the next word."

if __name__ == "__main__":
    app.run(debug=True)
