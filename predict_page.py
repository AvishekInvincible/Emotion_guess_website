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
    pred = model.predict([sentence])[0]
    st.markdown(""" 
        div.stButton > button:first-child {
        background-color: #00cc00;color:white;font-size:20px;height:3em;width:30em;border-radius:10px 10px 10px 10px;
        }
        .css-2trqyj:focus:not(:active) {
        border-color: #ffffff;
        box-shadow: none;
        color: #ffffff;
        background-color: #0066cc;
        }
        .css-2trqyj:focus:(:active) {
        border-color: #ffffff;
        box-shadow: none;
        color: #ffffff;
        background-color: #0066cc;
        }
        .css-2trqyj:focus:active){
        background-color: #0066cc;
        border-color: #ffffff;
        box-shadow: none;
        color: #ffffff;
        background-color: #0066cc;
        }
        """, unsafe_allow_html=True)
    if sentence:
        st.write("Model's prediction : ",pred)

    if st.button("Right"):
       df2 = pd.DataFrame([sentence, 1], columns=['Sentence','Pred'])
       df3 = df.append(df2)
       df3.to_csv('data.csv')
    if st.button("Wrong"):
       df2 = pd.DataFrame([sentence, 0], columns=['Sentence','Pred']) 
       df3 = df.append(df2)
       df3.to_csv('data.csv')
# conda activate ml
show_predict_page(df)
