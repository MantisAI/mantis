from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import boto3
import os
import numpy as np
import json

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


def MantisGutenbergMiniLM():
    s3 = boto3.client(
        's3', 
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    for doc_i in range(1,87):
        s3.download_file(
            "mantisnlp-kaggle-data",
            "minilm_embeddings_{}.json".format(doc_i),
            'minilm_embeddings.json'
        )
        with open("minilm_embeddings.json", 'r') as f:
            embedding_data = json.load(f)
            embedding_data['embeddings'] = np.array(embedding_data['embeddings'])
            return embedding_data

def get_gutenberg_text():
    s3 = boto3.client(
        's3', 
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    s3.download_file(
            "mantisnlp-kaggle-data",
            "all_gutenberg_text.txt",
            "all_gutenberg_text.txt"
        )


def construct_embedding_data(
        txt_file, 
        out_dir="./", 
        model_encoder_class=SentenceTransformerEncoder,
        character_length_threshold=100
    ):
    model_encoder = model_encoder_class()

    embedding_data = {
        "paragraphs": [],
        "embeddings": []
    }
    doc_current = 0
    doc_nr = 0

    with open(txt_file, 'r') as file:
        still_reading = True
        nr_paragraphs = 0
        while still_reading:
            line_i = 0
            paragraphs = []
            line = file.readline()
            while line and line_i<10000:
                if line and len(line) > character_length_threshold:
                    paragraphs.append(line.rstrip())
                    line_i += 1
                    nr_paragraphs += 1
                line = file.readline()

            if line_i<10000:
                still_reading = False

            embeddings = model_encoder.get_embeddings(paragraphs)
            
            embedding_data['paragraphs'].extend(paragraphs)
            embedding_data['embeddings'].extend(embeddings.tolist())
            doc_current+=1
            del embeddings

            if doc_current == 10 or still_reading is False:
                doc_nr += 1
                with open("{}/embeddings_{}.json".format(out_dir, doc_nr), 'w') as f:
                    json.dump(embedding_data, f)

                embedding_data = {
                    "paragraphs": [],
                    "embeddings": []
                }
                print(doc_nr)
                doc_current = 0


def get_similar_data(
        data, 
        labels=None, 
        threshold=0.7, 
        embeddings_generator=MantisGutenbergMiniLM, 
        model_encoder_class=SentenceTransformerEncoder
        ):
    model_encoder = model_encoder_class()
    data_embeddings = model_encoder.get_embeddings(data)

    generator = embeddings_generator()
    similar_data = []
    while True:
        embedding_data = next(generator)
        if embedding_data:
            scores = np.inner(data_embeddings, embedding_data['embeddings'])
            for i in range(0, scores.shape[0]):
                max_i = np.argmax(scores[i])
                if scores[i][max_i] >= threshold:
                    data_item = {"text": embedding_data['paragraphs'][max_i]}
                    if labels:
                        data_item['label']= labels[i]
                    similar_data.append(data_item)
        else:
            break

    return similar_data
