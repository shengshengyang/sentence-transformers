import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Read data from phone.xlsx
data = pd.read_excel('embed_query/phone2.xlsx')

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Load the Faiss index and title vectors from files
index = faiss.read_index('embed_query/index.faiss')
title_vectors = np.load('embed_query/title_vectors.npy')

# Set up Streamlit app
st.title("大豐智慧分機表")
query = st.text_input("請輸入您的問題，將為您找到合適的人:")
if query:
    # Vectorize the query
    query_vector = model.encode(query)

    # Search for nearest neighbors
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(np.array([query_vector]), k)

    # Retrieve the matched content
    matched_data = data.iloc[indices[0]]

    st.subheader("Matched Contents:")
    for i, row in matched_data.iterrows():
        st.write(f"{i + 1}. 部門: {row['dept']}")
        st.write(f"   姓名: {row['name']}")
        st.write(f"   分機: {row['ext']}")
        st.write(f"   私人手機: {row['privatePhone']}")
        st.write(f"   公務手機: {row['publicPhone']}")
        st.write(f"   手機簡碼65+分機3碼: {row['easyCode']}")
        st.write(f"   信箱: {row['email']}")
        st.write("---")
