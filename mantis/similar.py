from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import boto3
import os
import numpy as np
import json
import time

class MantisEncoder:
    def get_embeddings(self, data):
        """
        Parameters:
        data (list): List of strings

        Returns: 
        list: List of embeddings for the given data
        """
        pass

class SentenceTransformerEncoder(MantisEncoder):

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_embeddings(self, data):
        return self.model.encode(data)


class UniversalSentenceEncoder(MantisEncoder):

    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def get_embeddings(self, data):
        return self.model(data).numpy()


def MantisGutenbergMiniLM():
    """
    Generator that downloads the minilm embedding files one at a time 
    and for each yields the group of 100,000 vectors
    """

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
            yield embedding_data

def get_gutenberg_text():
    """
    Download the mantis hosted all_gutenberg_text.txt file (~5Gb)
    File contains paragraphs taken from ~15,000 open source books available on gutenberg.
    Each paragraph is on a separate line
    """

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
        character_length_threshold=100,
        verbose=True,
        vectors_per_file=100000
    ):
    """
    Based on selected model encoder, and given text data, constructs the vector embeddings of the data.

    Parameters:
    txt_tile (string): Filename of the textfile. File should have on each separate line the desired data
    out_dir (string): Directory name where the embedding json files will be created
    model_encoder_class (MantisEncoder): A interpretation of the MantisEncoder that implements the get_embeddings() function
    character_length_threshold (int): lower limit for nr of characters each item should have
    verbose (bool): If True then some log data will be printed
    vectors_per_file (int): How many embeddings one json file should contain. Depending on memory you can increase or decrease this

    Returns:
    Will create the embedding json files for the given text data. Each embedding file has 2 lists:
    paragraphs - the original text data
    embeddings - the embedding vector for the respective text item
    """

    model_encoder = model_encoder_class()
    vectors_per_chunk = int(vectors_per_file/10)

    embedding_data = {
        "paragraphs": [],
        "embeddings": []
    }
    doc_current = 0
    doc_nr = 0
    start_time = time.time()

    with open(txt_file, 'r') as file:
        still_reading = True
        nr_paragraphs = 0
        while still_reading:
            line_i = 0
            paragraphs = []
            line = file.readline()
            while line and line_i<vectors_per_chunk:
                if line and len(line) > character_length_threshold:
                    paragraphs.append(line.rstrip())
                    line_i += 1
                    nr_paragraphs += 1
                line = file.readline()

            if line_i<vectors_per_chunk:
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
                
                time_taken = time.time()-start_time
                start_time = time.time()
                embedding_data = {
                    "paragraphs": [],
                    "embeddings": []
                }
                doc_current = 0

                if verbose:
                    print("Processed set {} in {} seconds".format(doc_nr, time_taken))
    if verbose:
        print("Finished constructing embedding data")


def get_similar_data(
        data, 
        labels=None, 
        threshold=0.7, 
        embeddings_generator=MantisGutenbergMiniLM, 
        model_encoder_class=SentenceTransformerEncoder
        ):
    """
    Based on the data list, and the given generator, get a list of the most similar data that is above the given threshold

    Parameters:
    data (list): List of strings
    labels (list): List of labels. If given then the resulting similar data will have the label of the most similar in the original data
    threshold (float): Threshold of similarity. Increasing it will give less data but more similar, decreasing will give more but not as similar. Empirically we've found 0.7 seems a good middle value
    embeddings_generator (generator): Generator that yields sets of embedding dicts {paragraphs: list, embeddings: list}
    model_encoder_class (MantisEncoder): A interpretation of the MantisEncoder that implements the get_embeddings() function

    Returns:
    list of dict: The similar data in a list of dicts, if labels are present the dict will also include the 'label' key
    """

    model_encoder = model_encoder_class()
    data_embeddings = model_encoder.get_embeddings(data)

    generator = embeddings_generator()
    similar_data = []
    while True:
        embedding_data = next(generator)
        if embedding_data:
            scores = np.inner(embedding_data['embeddings'], data_embeddings)
            for i in range(0, scores.shape[0]):
                max_i = np.argmax(scores[i])
                if scores[i][max_i] >= threshold:
                    data_item = {"text": embedding_data['paragraphs'][i]}
                    if labels:
                        data_item['label']= labels[max_i]
                    similar_data.append(data_item)
        else:
            break

    return similar_data
