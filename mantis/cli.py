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
    embeddings_generator_name: str = "MantisGutenbergMiniLM",
    model_encoder_class_name: str = "SentenceTransformerEncoder"
):
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
