import json
import spacy
import typer
from wasabi import MarkdownRenderer

from .utils import create_prodigy_spans

from nervaluate.nervaluate import Evaluator


def predict_prodigy(
        model_path: str = typer.Argument(help="Path to spaCy model", default=""),
        data_path: str = typer.Argument(
            help="Path to data in prodigy format (including the raw text)",
            default=""
        ),
        tags: str = typer.Argument(
            None, help="Comma separated list of tags to include in the evaluation"
        ),
        by_tag: bool = typer.Option(
            None,
            help="If set, will return tag level results instead of aggregated results.",
        ),
        pretty: bool = typer.Option(
            None,
            help="If set, will print the results in a pretty format instead of returning the raw json",
        )
):
    spacy_model = spacy.load(model_path)

    true = []
    pred = []

    with open(data_path) as f:
        for line in f:
            pattern = json.loads(line)
            text = pattern["text"]
            meta = pattern["meta"]
            true.append(meta)
            doc = spacy_model(text)
            pred.append(create_prodigy_spans(doc))

    tags_list = tags.split(",")
    evaluator = Evaluator(true=true, pred=pred, tags=tags_list, loader=None)

    results, results_by_tag = evaluator.evaluate()

    if by_tag:
        output = results_by_tag
    else:
        output = results

    if pretty:
        md = MarkdownRenderer()
        md.add(md.code_block(output))
        typer.echo(md.text)
        return md.text
    else:
        typer.echo(output)
        return output
