from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict, Any, Mapping, Literal, Tuple

from .convert_functions import build_artifacts, save_raw_fif, write_snirf_h5
import numpy as np


PathLike = Union[str, Path]
Target = Literal["fif", "snirf", "both"]


@dataclass(frozen=True)
class ConvertConfig:
    """
    Optional advanced config.
    Any fields not None will override corresponding direct arguments in convert().
    """
    length_unit: Optional[str] = None        # "mm"|"cm"|"m"
    scale_to_m: Optional[bool] = None
    sfreq: Optional[float] = None            # None or <=0 -> infer in core
    store_mode: Optional[str] = None         # "hbo_hbr"|"intensity"
    no_plot: Optional[bool] = None
    export_csv: Optional[bool] = None
    verbose: Optional[bool] = None


@dataclass(frozen=True)
class ConvertResult:
    """
    Output paths. Either of them may be None depending on the target.
    """
    out_fif: Optional[Path] = None
    out_snirf: Optional[Path] = None


def _p(x: PathLike) -> Path:
    return x if isinstance(x, Path) else Path(str(x))


def _ensure_exists(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


def _set_verbose(verbose: bool) -> None:
    """
    Make verbose actually affect the same logger used in core.py:
        logger = logging.getLogger("snirf_converter")
    """
    import logging
    pkg_logger = logging.getLogger("snirf_converter")
    if verbose:
        pkg_logger.setLevel(logging.DEBUG)
        for h in pkg_logger.handlers:
            h.setLevel(logging.DEBUG)


def _validate(length_unit: str, store_mode: str) -> None:
    if length_unit not in ("mm", "cm", "m"):
        raise ValueError("length_unit must be one of: 'mm', 'cm', 'm'")
    if store_mode not in ("hbo_hbr", "intensity"):
        raise ValueError("store_mode must be 'hbo_hbr' or 'intensity'")


def _resolve_args(
    *,
    length_unit: str,
    scale_to_m: bool,
    sfreq: Optional[float],
    store_mode: str,
    no_plot: bool,
    export_csv: bool,
    verbose: bool,
    config: Optional[ConvertConfig],
) -> Tuple[str, bool, Optional[float], str, bool, bool, bool]:
    """
    Merge ConvertConfig overrides (non-None) into direct args.
    Returns resolved args in a stable order.
    """
    if config is not None:
        if config.length_unit is not None:
            length_unit = config.length_unit
        if config.scale_to_m is not None:
            scale_to_m = config.scale_to_m
        if config.sfreq is not None:
            sfreq = config.sfreq
        if config.store_mode is not None:
            store_mode = config.store_mode
        if config.no_plot is not None:
            no_plot = config.no_plot
        if config.export_csv is not None:
            export_csv = config.export_csv
        if config.verbose is not None:
            verbose = config.verbose

    _validate(length_unit=length_unit, store_mode=store_mode)
    _set_verbose(verbose)

    return length_unit, scale_to_m, sfreq, store_mode, no_plot, export_csv, verbose


def _run_pipeline(
    *,
    txt_path: PathLike,
    origin_path: PathLike,
    others_path: PathLike,
    subject: str,
    out_fif: Optional[PathLike],
    out_snirf: Optional[PathLike],
    events_path: Optional[PathLike],
    length_unit: str,
    scale_to_m: bool,
    sfreq: Optional[float],
    store_mode: str,
    no_plot: bool,
    export_csv: bool,
) -> ConvertResult:
    """
    Internal runner that validates files/dirs and calls core.build_artifacts + writers.
    """
    txt_p = _p(txt_path)
    ori_p = _p(origin_path)
    oth_p = _p(others_path)
    events_p = _p(events_path) if events_path is not None else None

    out_fif_p = _p(out_fif) if out_fif is not None else None
    out_snirf_p = _p(out_snirf) if out_snirf is not None else None

    _ensure_exists(txt_p, "TXT")
    _ensure_exists(ori_p, "origin CSV")
    _ensure_exists(oth_p, "others CSV")
    if events_p is not None:
        _ensure_exists(events_p, "events CSV")

    # Ensure output dirs exist
    if out_fif_p is not None:
        out_fif_p.parent.mkdir(parents=True, exist_ok=True)
    if out_snirf_p is not None:
        out_snirf_p.parent.mkdir(parents=True, exist_ok=True)

    # ---- build artifacts once ----
    art = build_artifacts(
        txt_path=str(txt_p),
        origin_path=str(ori_p),
        others_path=str(oth_p),
        events_path=str(events_p) if events_p else None,
        sfreq=sfreq,                 # None/<=0 -> infer inside core
        subject=subject,
        scale_to_m=scale_to_m,
        length_unit=length_unit,
    )

    # ---- optional preview (respect no_plot) ----
    if not no_plot:
        try:
            art.raw.plot(n_channels=min(40, len(art.raw.ch_names)), duration=30, show=True, block=True)
        except Exception:
            pass

    # ---- write FIF if requested ----
    if out_fif_p is not None:
        save_raw_fif(art.raw, str(out_fif_p), overwrite=True)

    # ---- write SNIRF if requested ----
    if out_snirf_p is not None:
        write_snirf_h5(
            out_path=str(out_snirf_p),
            time=art.time_arr,
            data_matrix=art.data_matrix,
            channel_pairs=art.channel_pairs,
            sources=art.sources_dict,
            detectors=art.detectors_dict,
            landmark_labels=list(art.origin_coords.keys()),
            landmark_coords=np.array(list(art.origin_coords.values())) if art.origin_coords else np.zeros((0, 3)),
            subject=subject,
            wavelengths=(760, 830),
            events_dict=art.events_dict_for_snirf,
        )

    return ConvertResult(out_fif=out_fif_p, out_snirf=out_snirf_p)


def convert(
    *,
    txt_path: PathLike,
    origin_path: PathLike,
    others_path: PathLike,
    out_fif: PathLike,
    subject: str,
    out_snirf: Optional[PathLike] = None,
    events_path: Optional[PathLike] = None,
    # direct args (CLI-like)
    length_unit: str = "mm",
    scale_to_m: bool = True,
    sfreq: Optional[float] = None,           # None or <=0 -> infer in core
    store_mode: str = "hbo_hbr",
    no_plot: bool = True,
    export_csv: bool = False,
    verbose: bool = False,
    # advanced override
    config: Optional[ConvertConfig] = None,
) -> ConvertResult:
    """
    Public API: Shimadzu TXT (+ coords, optional events) -> FIF and optional SNIRF.
    Always writes FIF; SNIRF is optional.
    """
    length_unit, scale_to_m, sfreq, store_mode, no_plot, export_csv, _ = _resolve_args(
        length_unit=length_unit,
        scale_to_m=scale_to_m,
        sfreq=sfreq,
        store_mode=store_mode,
        no_plot=no_plot,
        export_csv=export_csv,
        verbose=verbose,
        config=config,
    )

    return _run_pipeline(
        txt_path=txt_path,
        origin_path=origin_path,
        others_path=others_path,
        subject=subject,
        out_fif=out_fif,
        out_snirf=out_snirf,
        events_path=events_path,
        length_unit=length_unit,
        scale_to_m=scale_to_m,
        sfreq=sfreq,
        store_mode=store_mode,
        no_plot=no_plot,
        export_csv=export_csv,
    )


def convert_fif(
    *,
    txt_path: PathLike,
    origin_path: PathLike,
    others_path: PathLike,
    out_fif: PathLike,
    subject: str,
    events_path: Optional[PathLike] = None,
    # direct args (CLI-like)
    length_unit: str = "mm",
    scale_to_m: bool = True,
    sfreq: Optional[float] = None,
    store_mode: str = "hbo_hbr",
    no_plot: bool = True,
    export_csv: bool = False,
    verbose: bool = False,
    config: Optional[ConvertConfig] = None,
) -> ConvertResult:
    """
    Convert and write ONLY FIF.
    """
    length_unit, scale_to_m, sfreq, store_mode, no_plot, export_csv, _ = _resolve_args(
        length_unit=length_unit,
        scale_to_m=scale_to_m,
        sfreq=sfreq,
        store_mode=store_mode,
        no_plot=no_plot,
        export_csv=export_csv,
        verbose=verbose,
        config=config,
    )

    return _run_pipeline(
        txt_path=txt_path,
        origin_path=origin_path,
        others_path=others_path,
        subject=subject,
        out_fif=out_fif,
        out_snirf=None,
        events_path=events_path,
        length_unit=length_unit,
        scale_to_m=scale_to_m,
        sfreq=sfreq,
        store_mode=store_mode,
        no_plot=no_plot,
        export_csv=export_csv,
    )


def convert_snirf(
    *,
    txt_path: PathLike,
    origin_path: PathLike,
    others_path: PathLike,
    out_snirf: PathLike,
    subject: str,
    events_path: Optional[PathLike] = None,
    # direct args (CLI-like)
    length_unit: str = "mm",
    scale_to_m: bool = True,
    sfreq: Optional[float] = None,
    store_mode: str = "hbo_hbr",
    no_plot: bool = True,
    export_csv: bool = False,
    verbose: bool = False,
    config: Optional[ConvertConfig] = None,
) -> ConvertResult:
    """
    Convert and write ONLY SNIRF (HDF5-based).
    Note: main_pipeline must support out_fif=None.
    """
    length_unit, scale_to_m, sfreq, store_mode, no_plot, export_csv, _ = _resolve_args(
        length_unit=length_unit,
        scale_to_m=scale_to_m,
        sfreq=sfreq,
        store_mode=store_mode,
        no_plot=no_plot,
        export_csv=export_csv,
        verbose=verbose,
        config=config,
    )

    return _run_pipeline(
        txt_path=txt_path,
        origin_path=origin_path,
        others_path=others_path,
        subject=subject,
        out_fif=None,
        out_snirf=out_snirf,
        events_path=events_path,
        length_unit=length_unit,
        scale_to_m=scale_to_m,
        sfreq=sfreq,
        store_mode=store_mode,
        no_plot=no_plot,
        export_csv=export_csv,
    )


def convert_batch(
    jobs: Iterable[Mapping[str, Any]],
    *,
    target: Target = "both",
    # batch defaults (CLI-like)
    length_unit: str = "mm",
    scale_to_m: bool = True,
    sfreq: Optional[float] = None,
    store_mode: str = "hbo_hbr",
    no_plot: bool = True,
    export_csv: bool = False,
    verbose: bool = False,
    config: Optional[ConvertConfig] = None,
) -> List[ConvertResult]:
    """
    Batch conversion.

    jobs required keys:
      txt_path, origin_path, others_path, subject
    and depending on target:
      - target="fif"  -> out_fif required
      - target="snirf" -> out_snirf required
      - target="both" -> out_fif required, out_snirf optional (or required if you want)

    optional:
      events_path
    """
    length_unit, scale_to_m, sfreq, store_mode, no_plot, export_csv, _ = _resolve_args(
        length_unit=length_unit,
        scale_to_m=scale_to_m,
        sfreq=sfreq,
        store_mode=store_mode,
        no_plot=no_plot,
        export_csv=export_csv,
        verbose=verbose,
        config=config,
    )

    results: List[ConvertResult] = []
    base_required = ("txt_path", "origin_path", "others_path", "subject")

    for idx, j in enumerate(jobs):
        missing = [k for k in base_required if k not in j]
        if missing:
            raise KeyError(f"convert_batch job[{idx}] missing keys: {missing}")

        events_path_j = j.get("events_path")

        if target == "fif":
            if "out_fif" not in j:
                raise KeyError(f"convert_batch job[{idx}] missing key: out_fif (target='fif')")
            results.append(
                _run_pipeline(
                    txt_path=j["txt_path"],
                    origin_path=j["origin_path"],
                    others_path=j["others_path"],
                    subject=j["subject"],
                    out_fif=j["out_fif"],
                    out_snirf=None,
                    events_path=events_path_j,
                    length_unit=length_unit,
                    scale_to_m=scale_to_m,
                    sfreq=sfreq,
                    store_mode=store_mode,
                    no_plot=no_plot,
                    export_csv=export_csv,
                )
            )
        elif target == "snirf":
            if "out_snirf" not in j:
                raise KeyError(f"convert_batch job[{idx}] missing key: out_snirf (target='snirf')")
            results.append(
                _run_pipeline(
                    txt_path=j["txt_path"],
                    origin_path=j["origin_path"],
                    others_path=j["others_path"],
                    subject=j["subject"],
                    out_fif=None,
                    out_snirf=j["out_snirf"],
                    events_path=events_path_j,
                    length_unit=length_unit,
                    scale_to_m=scale_to_m,
                    sfreq=sfreq,
                    store_mode=store_mode,
                    no_plot=no_plot,
                    export_csv=export_csv,
                )
            )
        else:  # "both"
            if "out_fif" not in j:
                raise KeyError(f"convert_batch job[{idx}] missing key: out_fif (target='both')")
            results.append(
                _run_pipeline(
                    txt_path=j["txt_path"],
                    origin_path=j["origin_path"],
                    others_path=j["others_path"],
                    subject=j["subject"],
                    out_fif=j["out_fif"],
                    out_snirf=j.get("out_snirf"),
                    events_path=events_path_j,
                    length_unit=length_unit,
                    scale_to_m=scale_to_m,
                    sfreq=sfreq,
                    store_mode=store_mode,
                    no_plot=no_plot,
                    export_csv=export_csv,
                )
            )

    return results


__all__ = [
    "PathLike",
    "Target",
    "ConvertConfig",
    "ConvertResult",
    "convert",
    "convert_fif",
    "convert_snirf",
    "convert_batch",
]
