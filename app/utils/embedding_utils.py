from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
def embedding_predict(text, reference_embeddings):
    text_emb = embedding_model.encode([text])[0]

    scores = {}
    for label, embeds in reference_embeddings.items():
        score = cosine_similarity(
            [text_emb], embeds
        ).mean()
        scores[label] = score

    best_label = max(scores, key=scores.get)
    confidence = scores[best_label]

    return best_label, float(confidence)