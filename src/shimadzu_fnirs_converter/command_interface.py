from __future__ import annotations

import argparse
import logging
from typing import Optional, List

from .convert_functions import main_pipeline

# -------------------------
# Logger 配置（与 core.py 保持一致）
# core.py: logger = logging.getLogger("snirf_converter")
# -------------------------
logger = logging.getLogger("snirf_converter")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shimadzu_fnirs_converter",
        description=(
            "Convert Shimadzu TXT + origin/others (+ optional events) "
            "-> MNE Raw (.fif) and optional SNIRF (.snirf)."
        ),
    )

    # required
    p.add_argument("--txt", required=True, help="Shimadzu TXT (raw) file path")
    p.add_argument("--origin", required=True, help="Origin CSV path (Nz/AL/AR/Cz etc.)")
    p.add_argument("--others", required=True, help="Others CSV path (T*/R*/CH*)")
    p.add_argument("--out", required=True, help="Output MNE Raw .fif path")
    p.add_argument("--subject", required=True, help="Subject ID (e.g., sub-01)")

    # optional
    p.add_argument("--events", default=None, help="Optional events CSV (onset,duration,value[,condition])")
    p.add_argument("--snirf", default=None, help="Optional output SNIRF path")
    p.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency (Hz). If omitted or <=0, infer from TXT time column.",
    )

    # coords / scaling
    p.add_argument(
        "--length-unit",
        choices=["mm", "cm", "m"],
        default="mm",
        help="Unit used in coord CSV (default: mm)",
    )

    # ✅ 建议：默认开启 scale_to_m，和 API 默认一致
    p.add_argument(
        "--scale-to-m",
        action="store_true",
        default=True,
        help="Scale coordinates to meters (default: enabled).",
    )
    p.add_argument(
        "--no-scale-to-m",
        action="store_false",
        dest="scale_to_m",
        help="Do NOT scale coordinates to meters.",
    )

    # pipeline knobs
    p.add_argument(
        "--store-mode",
        choices=["hbo_hbr", "intensity"],
        default="hbo_hbr",
        help="How to store data in Raw (default: hbo_hbr)",
    )
    p.add_argument(
        "--export-csv",
        action="store_true",
        help="Export intermediate CSVs (matrix, chpos)",
    )

    # UX
    p.add_argument("--no-plot", action="store_true", help="Skip preview plotting")
    p.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG)")

    return p


def _set_verbose(verbose: bool) -> None:
    """
    Make --verbose actually affect the same logger used in core.py.
    Avoid touching root logger to prevent noisy dependencies.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
    else:
        # Let core.py decide default level/handlers; keep conservative here
        logger.setLevel(logging.INFO)
        for h in logger.handlers:
            h.setLevel(logging.INFO)


def main(argv: Optional[List[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    _set_verbose(args.verbose)

    # core.main_pipeline() 没有 verbose 参数：靠 logger 级别控制
    main_pipeline(
        txt_path=args.txt,
        origin_path=args.origin,
        others_path=args.others,
        events_path=args.events,
        out_fif=args.out,
        out_snirf=args.snirf,
        sfreq=args.sfreq,              # None/<=0 -> core 内部自动推断
        subject=args.subject,
        scale_to_m=args.scale_to_m,
        length_unit=args.length_unit,
        store_mode=args.store_mode,
        no_plot=args.no_plot,
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    main()
