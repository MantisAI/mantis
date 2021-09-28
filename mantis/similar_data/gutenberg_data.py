import boto3
import os
import numpy as np
import json

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