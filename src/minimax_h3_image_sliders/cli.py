"""Offline planning CLI. Native operation names fail without importing a runtime."""
import argparse
import json
from .contract import OPERATIONS, NativeUnavailable, RequestPlan, refuse_native, source_lock

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="emit metadata only; do not read reference files")
    plan.add_argument("--prompt", required=True)
    plan.add_argument("--width", type=int, default=4096)
    plan.add_argument("--height", type=int, default=4096)
    plan.add_argument("--reference", action="append", default=[])
    plan.add_argument("--seed", type=int, default=7)
    plan.add_argument("--strength", type=float, default=1.0)
    sub.add_parser("provenance", help="print the packaged inspection record")
    sub.add_parser("cpu-check", help="probe the actual shared projection on random tensors only")
    for name in OPERATIONS:
        sub.add_parser(name, help="unavailable in this source-only release")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = RequestPlan(args.prompt, args.width, args.height, tuple(args.reference), args.seed, args.strength).as_dict()
        elif args.command == "provenance":
            result = source_lock()
        elif args.command == "cpu-check":
            from .synthetic import check
            result = check()
        else:
            refuse_native(args.command)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, NativeUnavailable) as error:
        parser.exit(2, f"error: {error}\n")

if __name__ == "__main__":
    raise SystemExit(main())
