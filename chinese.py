import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import csv
import requests
import json

embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

corpus = []

# Read the CSV file and extract the corpus
with open('corpus.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        corpus.append(row[0])  # Assuming the text is in the first column of each row

# Check if embeddings are already saved
try:
    with open('corpus_embeddings.pkl', 'rb') as f:
        corpus_embeddings = pickle.load(f)
except FileNotFoundError:
    # Encode the corpus and save the embeddings
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    with open('corpus_embeddings.pkl', 'wb') as f:
        pickle.dump(corpus_embeddings, f)

# Streamlit app
st.title("大豐智慧分機表")

# User input
user_input = st.text_input("請問要找誰:")

if user_input:
    # Find the closest sentences in the corpus based on cosine similarity
    query_embedding = embedder.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    # Display the results
    st.subheader("五名與您問題最相近的人:")
    for score, idx in zip(top_results[0], top_results[1]):
        st.write(corpus[idx], "(Score: {:.4f})".format(score))
