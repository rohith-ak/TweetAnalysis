# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:07:51 2020
@author: win10
"""
#pip install fastapi uvicorn

# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
import pickle
import re
import numpy as np
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.backend import clear_session
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 2. Create the app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

def basic_cleaning(ip_text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', str(' '.join(x.lower() for x in str(ip_text).split())))
    return text
    
# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/Welcome')
def get_name(name: str):
    max_review_length = 300
    model_test = tf.keras.models.load_model('./model_v1.h5')
    tokenizer_object = pd.read_pickle(r'./tokenizer_tweet_sentiment_analysis')
    loaded_model = joblib.load('./labelEncoder.joblib')

    inputString = name#"best of how to (conceptually)"
    inputString = basic_cleaning(inputString)

    seq1=tokenizer_object.texts_to_sequences([inputString])
    seq2=seq1
    print(seq2)
    seq = pad_sequences(seq2,
                        maxlen=max_review_length,
                        padding='pre',  
                        truncating='post')
    print(seq)
    scores = model_test.predict(seq.reshape(1,-1), verbose=1)
    print(scores)
    y_classes = np.argmax(scores, axis=None, out=None)
    print(y_classes)
    print("".join(loaded_model.inverse_transform([y_classes])))
    #return (loaded_model.inverse_transform([y_classes]))
    return {'Sentiment Type': "".join(loaded_model.inverse_transform([y_classes]))}

#    return {'Welcome To LSTM demo': f'{name}'}



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
