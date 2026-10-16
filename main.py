"""Main entry point for the Hotel Booking ML project.

The command implementations will be added incrementally as the project develops.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hotel Booking Demand end-to-end ML project"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("analyze", help="Run data analysis / validation.")
    train = subparsers.add_parser("train", help="Train a model.")
    train.add_argument("--task", choices=["regression", "classification"], required=True)

    tune = subparsers.add_parser("tune", help="Tune a model.")
    tune.add_argument("--task", choices=["regression", "classification"], required=True)

    subparsers.add_parser("evaluate", help="Evaluate trained models.")
    subparsers.add_parser("cluster", help="Run clustering analysis.")

    predict = subparsers.add_parser("predict", help="Run inference.")
    predict.add_argument("--task", choices=["regression", "classification"], required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print(
        f"Command '{args.command}' is scaffolded but not implemented yet. "
        "Implementation will be added incrementally."
    )


if __name__ == "__main__":
    main()
