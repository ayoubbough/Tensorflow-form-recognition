import os
import shutil
from typing import Generator

import pytest

import mnist_keras

# mypy: ignore-errors

@pytest.fixture
def clean_model_dir() -> Generator[str, None, None]:
    path = "./model_keras_test"
    if os.path.exists(path):
        shutil.rmtree(path)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

@pytest.mark.integration
def test_integration_full_cycle(clean_model_dir: str) -> None:
    """
    Vérifie que le fichier modèle est réellement créé.
    """
    model, outputs = mnist_keras.train_model(output_dir=clean_model_dir, steps=10)
    
    assert os.path.exists(clean_model_dir)
    # TF2 SavedModel crée saved_model.pb et le répertoire variables
    assert os.path.exists(os.path.join(clean_model_dir, "saved_model.pb"))
