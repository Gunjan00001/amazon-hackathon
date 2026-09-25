import argparse

from ber.blocking import run_block
from ber.config import Config
from ber.prepare import run_prepare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ber")
    parser.add_argument(
        "command",
        choices=["prepare", "block", "audit", "features", "train", "tune", "predict", "outputs", "all"],
    )
    parser.add_argument("--config", default="code/business_entity_resolution/config.json")
    parser.add_argument("--split", default="both", choices=["train", "test", "both"])
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.command in ("prepare", "all"):
        print(run_prepare(cfg))
    if args.command in ("block", "all"):
        splits = ("train", "test") if args.split == "both" else (args.split,)
        for split in splits:
            print(run_block(cfg, split))
    if args.command in ("audit", "all"):
        from ber.audit import run_audit

        print(run_audit(cfg, "train"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
