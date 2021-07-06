import os
import json
import spacy
import typer

from .utils import create_prodigy_spans


def predict_prodigy(
        model_path: str = typer.Argument(help="Path to spaCy model", default=""),
        data_path: str = typer.Argument(
            help="Path to data in prodigy format (including the raw text)",
            default=""
        )
):
    spacy_model = spacy.load(model_path)

    pred = []

    with open(data_path) as f:
        for line in f:
            pattern = json.loads(line)
            text = pattern["text"]
            doc = spacy_model(text)
            pred.append({"text": text, "meta": create_prodigy_spans(doc)})

    head, tail = os.path.split(data_path)

    filename, extension = os.path.splitext(tail)
    pred_path = os.path.join(head, filename + "_pred" + extension)

    with open(pred_path, "w") as f:
        for prediction in pred:
            f.write(json.dumps(prediction))
            f.write('\n')
    return
