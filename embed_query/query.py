import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Read data from phone.json
with open('embed_query/phone.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Load the Faiss index and title vectors from files
index = faiss.read_index('embed_query/index.faiss')
title_vectors = np.load('embed_query/title_vectors.npy')

# Set up Streamlit app
st.title("Nearest Neighbor Search")
query = st.text_input("Enter your query:")
if query:
    # Vectorize the query
    query_vector = model.encode(query)

    # Search for nearest neighbors
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(np.array([query_vector]), k)

    # Retrieve the matched content as replies
    matched_contents = [data[index]['content'] for index in indices[0]]

    st.subheader("Matched Contents:")
    for i, content in enumerate(matched_contents):
        st.write(f"{i+1}. {content}")
