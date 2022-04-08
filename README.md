# bert-onnx-tensorrt-dynamic-batching

> **Era:** 2022 · technique from 2022, reimplemented in 2026 with current PyTorch, ONNX, and ONNX Runtime.

Scaffold for a small BERT-style encoder, genuine ONNX export and CPU execution, a dynamic-batching
server, latency/throughput measurements, and an explicitly pending TensorRT path.

## Results

CPU latency and throughput: **pending**. TensorRT export, execution, and metrics: **pending**.
No inference benchmark has been run yet.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest
```

## Limitations

Model export, runtime execution, and the batching server land on subsequent feature branches.
TensorRT is not installed and no TensorRT result is claimed.

## License

MIT
