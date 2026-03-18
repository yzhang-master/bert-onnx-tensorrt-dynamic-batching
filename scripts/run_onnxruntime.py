#!/usr/bin/env python3
import argparse

import numpy as np

from bert_dynamic_batching.runtime import OnnxEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to an exported ONNX encoder")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--sequence", type=int, default=11)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    ids = rng.integers(0, 128, size=(args.batch, args.sequence), dtype=np.int64)
    mask = np.ones_like(ids)
    output = OnnxEncoder(args.model).predict(ids, mask)
    print(f"ONNX Runtime CPU output: shape={output.shape}, dtype={output.dtype}")
    print(f"Output checksum: {float(output.sum()):.8f}")


if __name__ == "__main__":
    main()
