import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "full": [],
    "no_language": ["--disable-language"],
    "no_obstruction": ["--disable-obstruction"],
    "no_support": ["--disable-support"],
    "no_nearby": ["--disable-nearby"],
}


def main():
    parser = argparse.ArgumentParser(description="Launch controlled Graph Transformer policy ablations.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    output = Path(args.out_dir)
    for name, switches in VARIANTS.items():
        model_dir = output / name
        train_command = [sys.executable, "-m", "obstruction_train.train_graph_policy", "--train-jsonl", args.train_jsonl, "--out-dir", str(model_dir), "--epochs", str(args.epochs), "--seed", str(args.seed), "--device", args.device, *switches, *passthrough]
        eval_command = [sys.executable, "-m", "obstruction_train.evaluate_graph_policy", "--jsonl", args.val_jsonl, "--checkpoint", str(model_dir / "best.pt"), "--out", str(model_dir / "metrics.json"), "--device", args.device]
        print(" ".join(train_command), flush=True)
        print(" ".join(eval_command), flush=True)
        if not args.dry_run:
            subprocess.run(train_command, check=True)
            subprocess.run(eval_command, check=True)


if __name__ == "__main__":
    main()
