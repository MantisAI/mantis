from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub

class SentenceTransformerEncoder:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_embeddings(self, data):
        return self.model.encode(data)


class UniversalSentenceEncoder:

    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def get_embeddings(self, data):
        return self.model(data)