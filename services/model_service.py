import os
import pickle
from functools import lru_cache

MODELS_DIR = "models"
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.pkl")

# Fallback if model_results.pkl isn't there
DEFAULT_BEST_MODEL_FILE = "logistic_regression.pkl"

# Map your saved “best_model” label -> pkl filename
MODEL_FILE_MAP = {
    "Random Forest": "random_forest.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_model_bundle():
    """
    Returns a dict:
      {
        "model": <sklearn model>,
        "best_model_name": "Logistic Regression",
        "model_path": "models/logistic_regression.pkl",
        "results": {...}  # optional
      }

    Cached so it loads once per server run.
    """
    results = None
    best_name = None

    if os.path.exists(RESULTS_PATH):
        results = _load_pickle(RESULTS_PATH)
        best_name = results.get("best_model")

    # Decide which file to load
    if best_name in MODEL_FILE_MAP:
        model_file = MODEL_FILE_MAP[best_name]
    else:
        model_file = DEFAULT_BEST_MODEL_FILE
        if best_name is None:
            best_name = "Logistic Regression"  # sensible display name fallback

    model_path = os.path.join(MODELS_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py to generate .pkl files."
        )

    model = _load_pickle(model_path)

    return {
        "model": model,
        "best_model_name": best_name,
        "model_path": model_path,
        "results": results,
    }


def get_model():
    """Convenience wrapper if you only want the model object."""
    return get_model_bundle()["model"]


def reload_model():
    """Use if you retrain and want the API to pick up the new model without restarting."""
    get_model_bundle.cache_clear()
    return get_model_bundle()
