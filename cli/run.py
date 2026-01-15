from __future__ import annotations

import argparse
from typing import List, Optional

from core.defaults import DEFAULT_CAUSAL_LM


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="anima")
    sub = parser.add_subparsers(dest="cmd", required=True)

    api_p = sub.add_parser("api", help="Run FastAPI server")
    api_p.add_argument("--host", default="0.0.0.0")
    api_p.add_argument("--port", type=int, default=8000)

    smoke_p = sub.add_parser("smoke", help="Tiny extractor smoke test")
    smoke_p.add_argument("--model", default=DEFAULT_CAUSAL_LM)
    smoke_p.add_argument("--prompt", default="Hello")

    args = parser.parse_args(argv)

    if args.cmd == "api":
        import uvicorn

        uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)
    elif args.cmd == "smoke":
        from core.extractor import ActivationExtractor

        ex = ActivationExtractor(args.model)
        rows = ex.extract(args.prompt, max_new_tokens=4)
        print("tokens:", len(rows))


if __name__ == "__main__":
    main()
