from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from core.defaults import DEFAULT_CAUSAL_LM


def _default_api_port() -> int:
    return int(os.environ.get("ANIMA_API_PORT", "8010"))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="anima")
    sub = parser.add_subparsers(dest="cmd", required=True)

    api_p = sub.add_parser("api", help="Run FastAPI server")
    api_p.add_argument("--host", default="127.0.0.1")
    api_p.add_argument("--port", type=int, default=None)

    smoke_p = sub.add_parser("smoke", help="Tiny extractor smoke test")
    smoke_p.add_argument("--model", default=DEFAULT_CAUSAL_LM)
    smoke_p.add_argument("--prompt", default="Hello")

    train_p = sub.add_parser("train", help="Train Narratives-aligned probe")
    train_p.add_argument("--model", default=DEFAULT_CAUSAL_LM)
    train_p.add_argument("--narratives-root", required=True)
    train_p.add_argument(
        "--target-mode",
        default="pca",
        choices=["pca", "atlas", "tribe_distill", "blend"],
        help="pca/atlas: Narratives targets; tribe_distill/blend: same + tribe projection save",
    )
    train_p.add_argument("--roi-npz", default=None)

    train_text_p = sub.add_parser("train-text", help="Train GoEmotions text probe")
    train_text_p.add_argument("--model", default="distilgpt2")
    train_text_p.add_argument("--max-samples", type=int, default=1500)
    train_text_p.add_argument("--epochs", type=int, default=12)
    train_text_p.add_argument("--device", default="cpu")

    train_all_p = sub.add_parser("train-text-all", help="Train GoEmotions probes for CPU model tier")
    train_all_p.add_argument("--tier", default="cpu", choices=["cpu", "large", "all"])
    train_all_p.add_argument("--max-samples", type=int, default=None)

    validate_p = sub.add_parser("validate", help="Steering / hedge validation")
    validate_p.add_argument("--model", default=DEFAULT_CAUSAL_LM)
    validate_p.add_argument("--prompt", default="I am not sure, but perhaps it might be fine.")
    validate_p.add_argument("--volatility", action="store_true", help="Compare valence std none vs dampen")

    bench_p = sub.add_parser("benchmark", help="Run benchmark suite")
    bench_p.add_argument("--model", default=DEFAULT_CAUSAL_LM)
    bench_p.add_argument("--tiers", default="internal")

    boot_p = sub.add_parser("bootstrap", help="First-time install: data, probes, tests")
    boot_p.add_argument("--skip-train", action="store_true")
    boot_p.add_argument("--skip-tests", action="store_true")

    zoo_p = sub.add_parser("train-zoo", help="Train text+brain probes for OSS model tier")
    zoo_p.add_argument("--tier", default="cpu", choices=["cpu", "large", "all"])
    zoo_p.add_argument("--text-only", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "api":
        import uvicorn

        port = args.port if args.port is not None else _default_api_port()
        uvicorn.run("api.server:app", host=args.host, port=port, reload=False)
    elif args.cmd == "smoke":
        from core.extractor import ActivationExtractor

        ex = ActivationExtractor(args.model)
        rows = ex.extract(args.prompt, max_new_tokens=4)
        print("tokens:", len(rows))
        ex.cleanup()
    elif args.cmd == "train":
        from core.extractor import ActivationExtractor
        from probes.train import train_probe
        from probes.zoo_io import probe_slug

        ex = ActivationExtractor(args.model)
        slug = probe_slug(args.model)
        mode = args.target_mode
        if mode in ("tribe_distill", "blend"):
            mode = "pca"
        _, meta = train_probe(
            ex,
            args.narratives_root,
            slug,
            target_mode=mode,
            roi_npz_path=args.roi_npz,
        )
        if args.target_mode in ("tribe_distill", "blend"):
            meta["tribe_training_mode"] = args.target_mode
        print(meta)
        ex.cleanup()
    elif args.cmd == "train-text":
        from probes.train_text import train_text_probe
        from probes.zoo_io import probe_slug, save_probe_bundle

        os.environ.setdefault("ANIMA_FORCE_CPU", "1" if args.device == "cpu" else "0")
        probe, meta = train_text_probe(
            args.model,
            max_samples=args.max_samples,
            epochs=args.epochs,
            device=args.device,
        )
        path = save_probe_bundle(probe, probe_slug(args.model), meta, suffix="_text")
        print("Saved", path)
    elif args.cmd == "train-text-all":
        import subprocess
        import sys

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "scripts" / "train_text_zoo_all.py"),
            "--tier",
            args.tier,
        ]
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        subprocess.run(cmd, check=False)
    elif args.cmd == "validate":
        from core.extractor import ActivationExtractor
        from probes.linear_probe import AffectProbe
        from probes.validate import validate_volatility_ablation, validate_with_controls
        from probes.zoo_io import load_probe_into

        ex = ActivationExtractor(args.model)
        probe = AffectProbe(ex.hidden_dim, len(ex.layer_indices))
        load_probe_into(probe, args.model)
        if args.volatility:
            print(validate_volatility_ablation(ex, probe, args.prompt))
        else:
            print(validate_with_controls(ex, probe, args.prompt))
        ex.cleanup()
    elif args.cmd == "benchmark":
        from benchmarks.run_all import main as bench_main
        import sys

        sys.argv = ["run_all", "--model", args.model, "--tiers", args.tiers]
        bench_main()
    elif args.cmd == "bootstrap":
        import subprocess
        import sys

        cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "bootstrap.py")]
        if args.skip_train:
            cmd.append("--skip-train")
        if args.skip_tests:
            cmd.append("--skip-tests")
        raise SystemExit(subprocess.call(cmd))
    elif args.cmd == "train-zoo":
        import subprocess
        import sys

        cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "train_zoo_full.py"), "--tier", args.tier]
        if args.text_only:
            cmd.append("--text-only")
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
