import streamlit as st 
from joblib import load
import numpy as np
data = {'Sentence':[],
        'Pred':[]}
df = pd.DataFrame(data)
model = load('EmotionalGuess.joblib')
def show_predict_page():
    st.title('Emotion Guess :-')
    st.write("""### Enter the text you want to classify """)
    sentence = st.text_input('Input your sentence here:') 
    pred = model.predict([sentence])[0]
    if sentence:
        st.write("Model's prediction : ",pred)

    if st.button("Right"):
       df2 = pd.DataFrame([sentence, 1], columns=['Sentence','Pred'])
       df3 = df1.append(df2)
       df3.to_csv('Out.csv')
    if st.button("Wrong"):
       df2 = pd.DataFrame([sentence, 0], columns=['Sentence','Pred']) 
       df3 = df1.append(df2)
       df3.to_csv('Out.csv')
# conda activate ml
show_predict_page()
