import argparse
import sys
from pathlib import Path

from .core import embed, extract


def _parse_cli():
    parser = argparse.ArgumentParser(
        prog="ffsecret",
        description="Fourier-domain image watermarking (FFSecret)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------
    # embed sub-command
    # ------------------------------------------------------------
    p_embed = sub.add_parser("embed", help="Embed data into an image")
    p_embed.add_argument("input", type=Path, help="Cover image (input)")
    p_embed.add_argument("output", type=Path, help="Stego image (output)")
    p_embed.add_argument("payload", help="Data to hide. Prefix with @ to read from file.")
    p_embed.add_argument("--key", "-k", type=int, required=True, help="Secret integer key")
    p_embed.add_argument("--alpha", type=float, help="Embedding strength (default: 0.1 or preset)")
    p_embed.add_argument("--spread", type=int, help="Coefficient pairs per bit (default: 5 or preset)")
    p_embed.add_argument("--preset", choices=["default", "robust"], default="default", help="Parameter preset")

    # ------------------------------------------------------------
    # extract sub-command
    # ------------------------------------------------------------
    p_extract = sub.add_parser("extract", help="Extract data from an image")
    p_extract.add_argument("input", type=Path, help="Stego image")
    p_extract.add_argument("--key", "-k", type=int, required=True, help="Secret integer key")
    p_extract.add_argument("--alpha", type=float, help="Embedding strength (must match embed side)")
    p_extract.add_argument("--spread", type=int, help="Spread factor (must match embed side)")
    p_extract.add_argument("--preset", choices=["default", "robust"], default="default", help="Parameter preset")
    p_extract.add_argument("--output", "-o", type=Path, help="Write extracted bytes to file instead of stdout")

    return parser.parse_args()


def main():
    args = _parse_cli()

    def resolve_params(preset: str, alpha_opt, spread_opt):
        if preset == "robust":
            alpha_def, spread_def = 0.25, 3
        else:
            alpha_def, spread_def = 0.1, 5

        alpha_val = alpha_opt if alpha_opt is not None else alpha_def
        spread_val = spread_opt if spread_opt is not None else spread_def
        return alpha_val, spread_val

    if args.cmd == "embed":
        # Resolve payload
        if args.payload.startswith("@"):
            payload_path = Path(args.payload[1:])
            data = payload_path.read_bytes()
        else:
            data = args.payload.encode("utf-8")

        alpha_val, spread_val = resolve_params(args.preset, args.alpha, args.spread)

        embed(
            str(args.input),
            str(args.output),
            data,
            args.key,
            alpha=alpha_val,
            spread=spread_val,
        )
        print(f"[ffsecret] Embedded {len(data)} bytes into {args.output}")

    elif args.cmd == "extract":
        alpha_val, spread_val = resolve_params(args.preset, args.alpha, args.spread)

        data = extract(
            str(args.input),
            args.key,
            alpha=alpha_val,
            spread=spread_val,
        )
        if args.output:
            args.output.write_bytes(data)
            print(f"[ffsecret] Extracted {len(data)} bytes -> {args.output}")
        else:
            # Binary-safe write to stdout
            sys.stdout.buffer.write(data)


if __name__ == "__main__":
    main() 