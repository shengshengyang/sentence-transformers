from sentence_transformers import SentenceTransformer, util
import torch
import pickle

embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

corpus = ['aa 人資 0930050000', 'aa 工程師 02222233']

# Check if embeddings are already saved
try:
    with open('corpus_embeddings.pkl', 'rb') as f:
        corpus_embeddings = pickle.load(f)
        print("找到")
except FileNotFoundError:
    # Encode the corpus and save the embeddings
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    with open('corpus_embeddings.pkl', 'wb') as f:
        pickle.dump(corpus_embeddings, f)
        print("用算的")

# Query sentences:
queries = ['我想找人資']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))
