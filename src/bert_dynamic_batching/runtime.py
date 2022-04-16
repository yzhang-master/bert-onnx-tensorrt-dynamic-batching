"""CPU ONNX Runtime execution for dynamically shaped encoder inputs."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxEncoder:
    def __init__(self, model_path: str | Path, *, threads: int = 1):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int64)
        mask = np.asarray(attention_mask, dtype=np.int64)
        if ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have equal [batch, sequence] shapes"
            )
        return self.session.run(
            ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask}
        )[0]
