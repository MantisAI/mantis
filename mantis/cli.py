import typer

from mantis.predict import predict_prodigy

app = typer.Typer()

predict_app = typer.Typer()
predict_app.command("prodigy")(predict_prodigy)
app.add_typer(predict_app, name="predict")

if __name__ == "__main__":
    app()
