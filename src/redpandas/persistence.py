import pickle
from typing import Any

def save_pipeline(prep_instance: Any, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(prep_instance, f)

def load_pipeline(filepath: str) -> Any:
    with open(filepath, 'rb') as f:
        return pickle.load(f)
