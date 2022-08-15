import pandas
import streamlit as st 
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_model():
    return load('EmotionalGuess.joblib')
model = get_model()
def show_predict_page():
    st.title('Emotion Guess :-')
    st.write("""### Enter the text you want to classify """)
    sentence = st.text_input('Input your sentence here:') 

    if sentence:
        st.write("Model's prediction : ",model.predict([sentence])[0])
        st.write()
# conda activate ml
