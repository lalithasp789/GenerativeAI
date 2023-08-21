
from transformers import pipeline
import streamlit as st

st.title("Sentiment Analyzer")

def sentiment(inputvalue):

    classifier = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    result = (classifier(inputvalue))
    return result
    

with st.form("my_form"):
    input_text = st.text_area("Enter text:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        result  = sentiment(input_text)
        st.info(f"Sentiment is :  {result[0]['label']} and Score is :  {result[0]['score']}")


