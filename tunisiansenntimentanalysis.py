import pickle
import streamlit as st
import re
from scipy import sparse
from scipy.sparse import csr_matrix
import os
print(os.getcwd())

st.title("Tunisian Video Comment Extraction for Sentiment")
try:
    with open('vectorizer.pickle', 'rb') as f:
        vect = pickle.load(f)
    with open('bestmodelfortunisiancomment.pickle', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'vectorizer.pickle' or 'bestmodel.pickle' not found. Please ensure they are present in the same directory.")
    exit()
text = st.text_input("أعطيني رايك")
if text:  # Check if text is not empty
    tmp = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tmp = re.sub(r'\s+', ' ', tmp)  # Replace multiple whitespaces with a single space
    tmp = re.sub(r'\s+' + r'[^a-zA-Z]', '', tmp)  # remove single chrachter
    print("Preprocessed Text (Streamlit):", tmp)
    X = vect.transform([tmp]).toarray()
    try:
        pred = model.predict(X)
        st.success(pred)
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")