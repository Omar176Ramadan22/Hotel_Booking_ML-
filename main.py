import argparse
from pathlib import Path
from src.regression.models import MODELS


def build_parser():
    parser = argparse.ArgumentParser(description="Hotel Booking ML")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("analyze", help="Show dataset information.")
    for name in ("train", "tune", "evaluate", "refit"):
        command = commands.add_parser(name)
        command.add_argument(
            "--task",
            choices=["regression", "classification", "clustering"],
            required=True,
        )
        command.add_argument("--model")
        if name == "refit":
            command.add_argument("--output-dir", help="Refit artifact directory (default: artifacts/)")
    commands.add_parser("cluster")
    compare = commands.add_parser("compare")
    compare.add_argument("--task", choices=["regression"], required=True)
    results = commands.add_parser("results", help="Show recorded regression comparison without fitting or scoring.")
    results.add_argument("--task", choices=["regression"], required=True)
    predict = commands.add_parser("predict")
    predict.add_argument(
        "--task", choices=["regression", "classification"], required=True
    )
    predict.add_argument("--model")
    predict.add_argument("--stage", choices=["baseline", "tuned"], default="tuned")
    predict.add_argument(
        "--input", help="JSON file; omit to enter booking details interactively."
    )
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--task", choices=["regression"], default="regression")
    diagnose.add_argument("--model", default="ridge")
    diagnose.add_argument("--stage", choices=["baseline", "tuned"], default="tuned")
    diagnose.add_argument("--source", choices=["training", "cv"], default="training")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "analyze":
            from src.data.analysis import run_analyze

            run_analyze()
        elif args.command == "cluster":
            print("clustering: not implemented yet")
        elif args.task != "regression":
            print(f"{args.task}: not implemented yet")
        elif args.command == "predict":
            from src.inference.predict import run_predict

            if args.model == "all":
                parser.error("Select one model for prediction.")
            run_predict(args.model, args.input, args.stage)
        elif args.command == "diagnose":
            from src.regression.diagnostics import run_diagnostics, run_training_diagnostics

            command = run_training_diagnostics if args.source == "training" else run_diagnostics
            command(args.model, args.stage)
        elif args.command == "compare":
            from src.evaluation.comparison import run_compare

            run_compare()
        elif args.command == "results":
            from src.evaluation.comparison import show_results

            show_results()
        else:
            if args.model is None:
                parser.error("--model is required for regression")
            from src.regression.train import run_train, run_tune, run_evaluate, run_refit

            commands = {"train": run_train, "tune": run_tune, "evaluate": run_evaluate, "refit": run_refit}
            names = list(MODELS) if args.model == "all" else [args.model]
            # Batch training skips finalized experiments while refit can reproduce them.
            frozen = set()
            if args.model == "all" and args.command in {"train", "tune"}:
                from src.data.preparation import load_config

                frozen = set(load_config()["regression"].get("frozen_models", []))
            for name in names:
                if name not in MODELS:
                    parser.error(f"Unknown regression model: {name}")
                if args.command == "refit" and args.model == "all":
                    recipe = Path("outputs/results") / f"regression_{name}_best_params.json"
                    if not recipe.exists():
                        print(f"{name}: no best-parameters JSON; skipped.")
                        continue
                if name in frozen:
                    print(f"{name}: frozen; skipped.")
                    continue
                if args.command == "refit" and args.output_dir is not None:
                    commands[args.command](name, output_dir=args.output_dir)
                else:
                    commands[args.command](name)
    except (OSError, ValueError, EOFError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
