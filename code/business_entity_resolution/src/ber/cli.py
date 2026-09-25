import argparse

from ber.config import Config
from ber.prepare import run_prepare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ber")
    parser.add_argument(
        "command",
        choices=["prepare", "block", "audit", "features", "train", "tune", "predict", "outputs", "all"],
    )
    parser.add_argument("--config", default="code/business_entity_resolution/config.json")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.command in ("prepare", "all"):
        counts = run_prepare(cfg)
        print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
