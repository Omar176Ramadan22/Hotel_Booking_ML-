import json
import math
import platform
from importlib.metadata import version
from pathlib import Path


def best_params_path(model_name):
    return Path("outputs/results") / f"regression_{model_name}_best_params.json"


def reproduction_hint(model_name):
    if best_params_path(model_name).exists():
        return f"run: python main.py refit --task regression --model {model_name}"
    return f"run train/tune for model {model_name} first"


def plain_parameters(parameters):
    # Convert NumPy scalars to JSON-compatible values.
    return {name: value.item() if hasattr(value, "item") else value for name, value in parameters.items()}


def write_metadata(artifact, model_name, stage, pipeline, seed, training_rows):
    parameters = plain_parameters(pipeline.named_steps["model"].get_params(deep=False))
    # Strict JSON rejects NaN, so store it as text.
    for name, value in parameters.items():
        if isinstance(value, float) and not math.isfinite(value):
            parameters[name] = str(value)
    parameters["numeric_degree"] = pipeline.get_params()["preprocess__encode__num__polynomial__degree"]
    versions = {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")}
    versions["python"] = platform.python_version()
    details = {
        "model": model_name, "stage": stage, "parameters": parameters,
        "seed": seed, "training_rows": training_rows, "versions": versions,
    }
    Path(str(artifact) + ".meta.json").write_text(
        json.dumps(details, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8",
    )
