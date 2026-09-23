"""Participant 1 reproducible offline preparation, training and verification."""
from __future__ import annotations

import argparse
import json
import sys

from windops.core import BackendError, data_root
from windops.ui.adapter import AdapterError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--interval-label", choices=("start", "end"), default="start")
    prepare_parser.add_argument("--scada-delay-minutes", type=int, default=0)
    prepare_parser.add_argument("--timestamp-format", default="mixed", help="pandas format; mixed uses day-first parsing")
    for command in ("baseline", "select", "backtest", "train-final", "export-february", "verify-baseline", "verify-backend"):
        commands.add_parser(command)
    args = parser.parse_args(argv)
    root = data_root()
    try:
        if args.command == "prepare":
            from .data import LabelPolicy, prepare
            if args.scada_delay_minutes < 0:
                parser.error("--scada-delay-minutes must be nonnegative")
            result = prepare(root, LabelPolicy(args.interval_label, args.scada_delay_minutes, args.timestamp_format))
        elif args.command.startswith("verify-") or args.command == "export-february":
            from .integration import export_february, verify_backend
            result = export_february(root) if args.command == "export-february" else verify_backend(root, baseline_only=args.command == "verify-baseline")
        else:
            from .training import baseline_stage, january_backtest, select, train_final
            result = {"baseline": baseline_stage, "select": select, "backtest": january_backtest, "train-final": train_final}[args.command](root)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except BackendError as exc:
        print(json.dumps({"status": "error", "error": exc.as_dict()}, ensure_ascii=False), file=sys.stderr)
        return 1
    except AdapterError as exc:
        print(json.dumps({"status": "error", "error": {"code": "ML_BACKEND_VERIFICATION", "message": str(exc), "retryable": False}},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
