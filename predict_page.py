import streamlit as st 
from joblib import load
import numpy as np


model = load('EmotionalGuess.joblib')
# def show_predict_page():
#     st.title('Emotion Guess :-')
#     st.write("""### Enter the text you want to classify """)
#     sentence = st.text_input('Input your sentence here:') 

#     if sentence:
#         st.write("Model's prediction : ",model.predict([sentence])[0])
# conda activate ml
# show_predict_page()
