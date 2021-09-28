import json
import numpy as np

from gutenberg_data import MantisGutenbergMiniLM, get_gutenberg_text
from models import SentenceTransformerEncoder


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
