import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def clean_and_reinput(data_path, model_path, index_path, vectors_path):
    try:
        # Read data from phone.json
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Initialize SentenceTransformer model
        model = SentenceTransformer(model_path)

        # Vectorize the titles
        title_vectors = []
        for item in data:
            title_vector = model.encode(item['title'])
            title_vectors.append(title_vector)

        # Convert title_vectors to a NumPy array
        title_vectors = np.array(title_vectors)

        # Initialize Faiss index
        index = faiss.IndexFlatL2(title_vectors.shape[1])

        # Add title vectors to the index
        index.add(title_vectors)

        # Save the index and title vectors to files
        faiss.write_index(index, index_path)
        np.save(vectors_path, title_vectors)

        print("Data cleaning and re-input successful.")
    except Exception as e:
        print("Error occurred during data cleaning and re-input:", str(e))


# Example usage
data_path = 'phone.json'
model_path = 'distiluse-base-multilingual-cased-v2'
index_path = 'index.faiss'
vectors_path = 'title_vectors.npy'

clean_and_reinput(data_path, model_path, index_path, vectors_path)
