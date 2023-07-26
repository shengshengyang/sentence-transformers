import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
import os
from dotenv import load_dotenv

# Read data from phone.xlsx
data = pd.read_excel('embed_query/phone2.xlsx')

# Load the Faiss index and title vectors from files
index = faiss.read_index('embed_query/index.faiss')
title_vectors = np.load('embed_query/title_vectors.npy')

# Set up Streamlit app
st.title("大豐智慧分機表")
query = st.text_input("請輸入您的問題，將為您找到合適的人:")
load_dotenv()
if query:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("OPENAI_KEY")}'  # Use the environment variable here
    }
    query_data = {
        "model": "text-embedding-ada-002",
        "input": [query]
    }
    response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=query_data)
    response_data = response.json()
    query_vector = np.array(response_data['data'][0]['embedding'])

    # Search for nearest neighbors
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(np.array([query_vector]), k)

    # Retrieve the matched content
    matched_data = data.iloc[indices[0]]

    st.subheader("Matched Contents:")
    for i, row in matched_data.iterrows():
        html = """
        <div style="border:1px solid #000; margin:10px; padding:10px;">
            <h2 style="color:#ff0000;">部門: {dept}</h2>
            <p>姓名: {name}</p>
            <p>分機: {ext}</p>
            <p>私人手機: {privatePhone}</p>
            <p>公務手機: {publicPhone}</p>
            <p>手機簡碼65+分機3碼: {easyCode}</p>
            <p>信箱: {email}</p>
        </div>
        """.format(index=i + 1, dept=row['dept'], name=row['name'], ext=row['ext'], privatePhone=row['privatePhone'],
                   publicPhone=row['publicPhone'], easyCode=row['easyCode'], email=row['email'])
        st.markdown(html, unsafe_allow_html=True)