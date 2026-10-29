"""CLI: `python -m fraud_alert_explainer train|explain`."""

import argparse
import logging

from fraud_alert_explainer import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraud_alert_explainer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="build features, train both models, write metrics and charts")
    explain = sub.add_parser("explain", help="explain test alerts with local LLMs and score them")
    explain.add_argument("--models", nargs="+", default=["qwen2.5:1.5b", "llama3.2:3b"])
    explain.add_argument("--n-alerts", type=int, default=40)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "train":
        pipeline.train()
    else:
        print(pipeline.explain(args.models, args.n_alerts))


if __name__ == "__main__":
    main()
