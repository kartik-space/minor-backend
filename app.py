from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import numpy as np
from PIL import Image
from pickle import load
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the tokenizer and model
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
max_length = 32

def extract_features_test(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")

    image = image.resize((299, 299))
    image = np.array(image)


    if image.shape[2] == 4:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def clearCaption(description):
    query = description
    stopwords = ['start', 'end']
    querywords = query.split()
    resultwords = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        photo = extract_features_test(img_path, xception_model)
        description = generate_desc(model, tokenizer, photo, max_length)
        description = clearCaption(description)
    return render_template("index.html", prediction=description, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
