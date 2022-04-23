import numpy as np
import onnx
import pytest
import torch

from bert_dynamic_batching import BertStyleEncoder, OnnxEncoder, export_encoder


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory):
    torch.manual_seed(7)
    model = BertStyleEncoder().eval()
    path = export_encoder(model, tmp_path_factory.mktemp("onnx") / "encoder.onnx")
    return model, path, OnnxEncoder(path)


def _inputs(batch: int, sequence: int):
    rng = np.random.default_rng(100 + batch + sequence)
    ids = rng.integers(0, 128, size=(batch, sequence), dtype=np.int64)
    mask = np.ones_like(ids)
    for row in range(batch):
        visible = max(1, sequence - row * 2 - 1)
        mask[row, visible:] = 0
    return ids, mask


def test_export_has_dynamic_batch_and_sequence_axes(exported_model) -> None:
    _, path, _ = exported_model
    graph = onnx.load(path).graph
    for value in [*graph.input, *graph.output]:
        dimensions = value.type.tensor_type.shape.dim
        assert dimensions[0].dim_param == "batch"
        assert dimensions[1].dim_param == "sequence"


@pytest.mark.parametrize("shape", [(1, 7), (3, 11), (2, 16)])
def test_onnxruntime_matches_pytorch_on_dynamic_shapes(shape, exported_model) -> None:
    model, _, runtime = exported_model
    ids, mask = _inputs(*shape)
    with torch.no_grad():
        expected = model(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
    actual = runtime.predict(ids, mask)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_padding_is_zero_and_cannot_affect_visible_tokens(exported_model) -> None:
    _, _, runtime = exported_model
    ids, mask = _inputs(2, 13)
    baseline = runtime.predict(ids, mask)
    assert np.count_nonzero(baseline[mask == 0]) == 0
    changed_ids = ids.copy()
    changed_ids[mask == 0] = 127
    changed = runtime.predict(changed_ids, mask)
    np.testing.assert_allclose(changed[mask != 0], baseline[mask != 0], atol=0, rtol=0)


def test_runtime_rejects_mismatched_arrays(exported_model) -> None:
    _, _, runtime = exported_model
    with pytest.raises(ValueError, match="equal"):
        runtime.predict(np.ones((2, 5), dtype=np.int64), np.ones((2, 4), dtype=np.int64))
