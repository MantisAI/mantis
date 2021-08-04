import os
import pytest

from mantis.predict import predict_prodigy


@pytest.fixture
def predict_prodigy(tmp_path):
    model_path = os.path.join(tmp_path, "en_core_web_lg")
    data_path = os.path.join(tmp_path, "data_path.json")

    predict_prodigy(model_path=model_path, data_path=data_path)
