"""Command-line entry point: ``fraud-mlops <generate|train|gate|serve|drift|export>``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from fraud_mlops.config import get_settings


def _cmd_generate(args: argparse.Namespace) -> int:
    from fraud_mlops.data import generate_transactions, save

    df = generate_transactions(
        n_rows=args.rows, fraud_rate=args.fraud_rate, seed=args.seed, start=args.start, drift=args.drift
    )
    path = save(df, args.out)
    print(json.dumps({"path": str(path), "rows": len(df), "fraud_rate": round(float(df.is_fraud.mean()), 5)}))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from fraud_mlops.data import load
    from fraud_mlops.train import summarize, train

    result = train(
        load(args.data),
        get_settings(),
        threshold_method=args.threshold_method,
        use_smote=args.smote,
        seed=args.seed,
        run_name=args.run_name,
    )
    print(json.dumps(summarize(result), indent=2))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from fraud_mlops.data import load
    from fraud_mlops.gate import run_gate

    decision = run_gate(load(args.data), get_settings(), candidate_version=args.version, report_path=args.report)
    print(json.dumps(decision.to_dict(), indent=2))
    return 0 if decision.promote or not args.strict else 1


def _cmd_drift(args: argparse.Namespace) -> int:
    from fraud_mlops.data import load
    from fraud_mlops.drift import drift_report, write_report

    settings = get_settings()
    report = drift_report(load(args.reference), load(args.current), psi_threshold=settings.psi_retrain_threshold)
    write_report(report, args.out)
    summary = {
        "report": args.out,
        "retrain_recommended": report["retrain_recommended"],
        "drifted_features": report["drifted_features"],
        "psi": {k: v["psi"] for k, v in report["features"].items()},
    }
    print(json.dumps(summary, indent=2))
    return 2 if (report["retrain_recommended"] and args.fail_on_drift) else 0


def _cmd_export(args: argparse.Namespace) -> int:
    from fraud_mlops.registry import export_bundle, load_alias, setup_mlflow

    settings = get_settings()
    bundle = load_alias(setup_mlflow(settings), settings.model_name, settings.champion_alias)
    out = export_bundle(bundle, args.out)
    print(json.dumps({"path": str(out), "version": bundle.version, "threshold": bundle.threshold}))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from fraud_mlops.serve import main

    main(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fraud-mlops", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic transaction dataset")
    g.add_argument("--rows", type=int, default=50_000)
    g.add_argument("--fraud-rate", type=float, default=0.015)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--start", default="2025-01-01")
    g.add_argument("--drift", type=float, default=0.0, help="0..1 population-shift intensity")
    g.add_argument("--out", default="data/transactions.parquet")
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="train, evaluate and register a candidate model")
    t.add_argument("--data", default="data/transactions.parquet")
    t.add_argument("--threshold-method", choices=["precision", "cost"], default="precision")
    t.add_argument("--smote", action="store_true", help="oversample the minority class with SMOTE")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--run-name", default=None)
    t.set_defaults(func=_cmd_train)

    gt = sub.add_parser("gate", help="compare candidate vs champion and promote if better")
    gt.add_argument("--data", default="data/transactions.parquet")
    gt.add_argument("--version", default=None, help="candidate version (default: latest)")
    gt.add_argument("--report", default="reports/gate_report.json")
    gt.add_argument("--strict", action="store_true", help="exit 1 when the candidate is rejected")
    gt.set_defaults(func=_cmd_gate)

    d = sub.add_parser("drift", help="PSI/KS drift report between reference and current data")
    d.add_argument("--reference", default="data/transactions.parquet")
    d.add_argument("--current", default="data/current.parquet")
    d.add_argument("--out", default="reports/drift_report.json")
    d.add_argument("--fail-on-drift", action="store_true", help="exit 2 when retraining is recommended")
    d.set_defaults(func=_cmd_drift)

    e = sub.add_parser("export", help="export the champion to a registry-independent bundle")
    e.add_argument("--out", default="artifacts/champion")
    e.set_defaults(func=_cmd_export)

    s = sub.add_parser("serve", help="run the FastAPI scoring service")
    s.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
