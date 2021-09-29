import typer
import sys
import pandas as pd

from mantis.predict import predict_prodigy
from mantis.similar import get_similar_data, MantisGutenbergMiniLM, SentenceTransformerEncoder, construct_embedding_data, UniversalSentenceEncoder

app = typer.Typer()

predict_app = typer.Typer()
predict_app.command("prodigy")(predict_prodigy)
app.add_typer(predict_app, name="predict")


@app.command()
def make_embedding_data(
    txt_file: str,
    out_dir: str = "./",
    character_length_threshold: int = 0.7,
    model_encoder_class_name: str = "SentenceTransformerEncoder"
):
    """
    This will call the construct_embedding_data. 

    Parameters:
    txt_tile (string): Filename of the textfile. File should have on each separate line the desired data
    out_dir (string): Directory name where the embedding json files will be created
    character_length_threshold (int): lower limit for nr of characters each item should have
    model_encoder_class_name (string): If you want to use a custom model, just implement the the MantisEncoder and make it available in the classpath, then you can specify the name here
    """

    model_encoder_class = SentenceTransformerEncoder
    if model_encoder_class_name != "SentenceTransformerEncoder":
        model_encoder_class = getattr(sys.modules[__name__], model_encoder_class_name)

    construct_embedding_data(
        txt_file, 
        out_dir=out_dir, 
        model_encoder_class=model_encoder_class,
        character_length_threshold=character_length_threshold
    )
    

@app.command()
def get_similar_training_data(
    data_csv: str = "clrp_train.csv",
    out_csv: str = "similar_data.csv",
    has_labels: bool = True,
    threshold: float = 0.7,
    model_encoder_class_name: str = "SentenceTransformerEncoder",
    embeddings_generator_name: str = "MantisGutenbergMiniLM"
):
    """
    Based on a given csv file this will construct the similar dataset and save it the out_csv file.

    Parameters:
    data_csv (string): Filename that contains the data you want to get similar data of. Should have 'text' column and if you want labels copied then also 'label' column
    out_csv (string): Filename that you want for the resulting data. Will have same format as the data_csv
    has_labels (bool): If True then labels will also be copied
    threshold (float): Threshold of similarity. Increasing it will give less data but more similar, decreasing will give more but not as similar. Empirically we've found 0.7 seems a good middle value
    model_encoder_class_name (string): If you want to use a custom model, just implement the the MantisEncoder and make it available in the classpath, then you can specify the name here
    embeddings_generator_name (string): This is similar to the model_encoder_class_name, but for the generator that gives the embeddings. By default our MantisGutenbergMiniLM will be used
    """

    data = pd.read_csv(data_csv)
    data_items = data['text']
    data_labels = None
    if has_labels:
        data_labels = data['label']
        
    embeddings_generator = MantisGutenbergMiniLM
    if embeddings_generator_name != "MantisGutenbergMiniLM":
        embeddings_generator = getattr(sys.modules[__name__], embeddings_generator_name)
    model_encoder_class = SentenceTransformerEncoder
    if model_encoder_class_name != "SentenceTransformerEncoder":
        model_encoder_class = getattr(sys.modules[__name__], model_encoder_class_name)
    
    similar_data = get_similar_data(
        data_items, 
        labels=data_labels, 
        threshold=threshold,
        embeddings_generator=embeddings_generator,
        model_encoder_class=model_encoder_class
    )

    similar_data_df = pd.DataFrame(similar_data)
    similar_data_df.to_csv(out_csv)

if __name__ == "__main__":
    app()
