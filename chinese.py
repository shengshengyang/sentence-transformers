import streamlit as st
from sympy import false

from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import csv
import requests
import json

embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

corpus = []

# Read the CSV file and extract the corpus
with open('corpus.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 0:
            corpus.append(row[0])

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
    top_results = torch.topk(cos_scores, k=3)
    # Convert top_results to a string
    top_results_str = " ".join(
        [str(corpus[idx]+",") for score, idx in zip(top_results[0], top_results[1])])

    print(top_results_str)
    # 用預設的板模去訪問chatgpt API
    # ChatGPT API endpoint
    api_endpoint = "https://api.openai.com/v1/chat/completions"
    # Generate response using ChatGPT API
    response = requests.post(
        api_endpoint,
        headers={
            "Authorization": "Bearer ",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "以下為參考資料" + top_results_str},
                {"role": "assistant", "content": "您好，請問需要我為這些資料做什麼?"},
                {"role": "user", "content": "請根據提供的參考資料，回答以下問題應該要找哪一位人員回答:" + user_input
                                            + ",若資料沒有能夠回答問題請回復: 目前尚無資料，請洽客服"}
            ],
            "temperature": 1,
            "top_p": 1,
            "n": 1
        }
    )
    # Check if 'choices' key exists in the response
    if 'choices' in response.json():
        # Extract the generated response
        generated_response = response.json()["choices"][0]["message"]["content"]

        # Print the generated response
        print(generated_response)
        st.write(generated_response)
    else:
        print("No response choices found.")


    # Display the results
    st.subheader("三名與您問題最相近的人:")
    for score, idx in zip(top_results[0], top_results[1]):
        st.write(corpus[idx], "(Score: {:.4f})".format(score))


