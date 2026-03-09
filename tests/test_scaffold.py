from pathlib import Path

import tomllib

import bert_dynamic_batching

ROOT = Path(__file__).parents[1]


def test_package_version_matches_project() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert bert_dynamic_batching.__version__ == project["version"]
    assert any(dep.startswith("onnx>=") for dep in project["dependencies"])
    assert any(dep.startswith("onnxruntime>=") for dep in project["dependencies"])


def test_tensorrt_is_an_optional_unclaimed_path() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "TensorRT export, execution, and metrics: **pending**" in readme
