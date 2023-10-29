import streamlit as st
import joblib
import string
import re
import spacy
import time

# @st.cache_resource(ttl=3600)
def load_model():
    learn_inf = joblib.load('checkpoints/spam_detection_model.pkl')
    vectorizer = joblib.load('checkpoints/count_vectorizer.pkl')
    return learn_inf,vectorizer


def clean_text(s): 
    for cs in s:
        if  not cs in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.rstrip('\r\n')

def remove_little(s): 
    wordsList = s.split()
    k_length=2
    resultList = [element for element in wordsList if len(element) > k_length]
    resultString = ' '.join(resultList)
    return resultString

def lemmatize_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

def preprocess(text):
    return lemmatize_text(remove_little(clean_text(text)))

def classify_email(model,vectorizer,email):
    prediction = model.predict(vectorizer.transform([email]))
    return prediction

def main():
    st.title("Spam Email Detector")
    output = st.empty()

    with st.spinner('Loading the webpage...'):
        user_input = st.text_area('Enter the email text:', '')
    
    if st.button("Check for Spam"):
        output.empty()
        if user_input:
            with st.status("Loading the model.....", expanded=True) as status:
                model,vectorizer = load_model()
                time.sleep(2)

                st.write("Analyzing the email.....")
                user_input = preprocess(user_input)
                time.sleep(2)

                st.write("Checking for Spam.....")
                prediction = classify_email(model,vectorizer,user_input)
                time.sleep(2)

                status.update(label="Detection Completed!", state="complete", expanded=False)


            if prediction == 1:
                output.error('Spam Detected!')
            else:
                output.success('Not Spam.')
        else:
            output.warning("Kindly enter the text to detect !!")

if __name__ == "__main__":
    main()