#!/usr/bin/env python3
import argparse

from bert_dynamic_batching.export import export_encoder
from bert_dynamic_batching.model import BertStyleEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/bert_style.onnx")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    import torch

    torch.manual_seed(args.seed)
    path = export_encoder(BertStyleEncoder(), args.output)
    print(f"Exported and validated {path}")


if __name__ == "__main__":
    main()
