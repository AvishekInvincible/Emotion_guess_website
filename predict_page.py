import streamlit as st 
from joblib import load
import numpy as np
import pandas as pd


df = pd.read_csv('data.csv')
model = load('EmotionalGuess.joblib')
def show_predict_page(df):
    st.title('Emotion Guess :-')
    st.write("""### Enter the text you want to classify """)
    sentence = st.text_input('Input your sentence here:')
    print(sentence)
    pred = model.predict([sentence])[0]
    data_r  = {'Sentence':sentence,'Pred':1}
    data_w  = {'Sentence':sentence,'Pred':0}

    if sentence:
        st.write("Model's prediction : ",pred)
    if st.button("Right"):
       df = df.append(data_r, ignore_index=True)
       df.to_csv('data.csv')
    elif st.button("Wrong"):
       df = df.append(data_w ,ignore_index=True)
       df.to_csv('data.csv')
# conda activate ml
show_predict_page(df)
