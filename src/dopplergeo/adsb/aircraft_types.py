"""Engine class of an ICAO aircraft type designator, and the jet filter.

The Doppler pipeline hears propeller aircraft, so jets are removed from the
candidate tracks. The policy is deliberately asymmetric: a track is dropped
only when its type designator is a KNOWN jet. Missing, blank, or unknown
designators are kept, because the type table is incomplete and a wrong
"jet" label would silently delete the correct answer.

The built-in table lives in :mod:`dopplergeo.adsb._typecodes` (a curated
subset of ICAO Doc 8643). :func:`load_type_table` extends it at runtime from
either the OpenSky aircraft database CSV or a two-column CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ._typecodes import TYPECODES as _BUILTIN

EngineClass = Literal["jet", "turboprop", "piston", "electric", "unknown"]

_CLASSES: frozenset[str] = frozenset({"jet", "turboprop", "piston", "electric"})

# Engine-type letter of an ICAO aircraft description such as "L2J" -> class.
_ENGINE_LETTER: dict[str, str] = {
    "J": "jet",
    "T": "turboprop",
    "P": "piston",
    "E": "electric",
}

# Live table. Starts as a copy of the built-in one; load_type_table extends it.
_TABLE: dict[str, str] = dict(_BUILTIN)


def _norm(typecode: object) -> str:
    """Upper-case, stripped designator; "" for None/NaN/non-strings."""
    if typecode is None:
        return ""
    if isinstance(typecode, float):  # NaN from a float column
        return ""
    s = str(typecode).strip().upper()
    if s in ("NAN", "<NA>", "NONE"):
        return ""
    return s


def engine_class(typecode: object) -> EngineClass:
    """Return "jet" | "turboprop" | "piston" | "electric" | "unknown"."""
    code = _norm(typecode)
    if not code:
        return "unknown"
    return _TABLE.get(code, "unknown")  # type: ignore[return-value]


def is_known_jet(typecode: object) -> bool:
    return engine_class(typecode) == "jet"


def filter_out_jets(tracks: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``typecode`` is a known jet. Unknown/blank types are kept.

    Returns a new frame with a fresh index. A frame without a ``typecode``
    column is returned unchanged (nothing is known, so nothing is dropped).
    """
    if "typecode" not in tracks.columns or len(tracks) == 0:
        return tracks.reset_index(drop=True)
    codes = tracks["typecode"]
    lookup = {c: is_known_jet(c) for c in codes.dropna().unique()}
    jet = codes.map(lookup).fillna(False).astype(bool)
    return tracks.loc[~jet].reset_index(drop=True)


def reset_type_table() -> None:
    """Discard anything loaded with :func:`load_type_table`."""
    _TABLE.clear()
    _TABLE.update(_BUILTIN)


def table_size() -> int:
    return len(_TABLE)


def _class_from_description(desc: str) -> str:
    """Engine class from an ICAO aircraft description like "L2J" or "H1T".

    Only landplanes, seaplanes and amphibians ("L", "S", "A") are mapped;
    helicopters, gyrocopters and tiltrotors return "unknown" because their
    "T" means turboshaft, which is not what the filter is about.
    """
    d = desc.strip().upper()
    if len(d) < 3 or d[0] not in ("L", "S", "A"):
        return "unknown"
    return _ENGINE_LETTER.get(d[-1], "unknown")


def _class_from_token(token: object) -> str:
    """Accept "jet"/"turboprop"/..., a single letter J/T/P/E, or "L2J"."""
    s = _norm(token)
    if not s:
        return "unknown"
    low = s.lower()
    if low in _CLASSES or low == "unknown":
        return low
    if len(s) == 1:
        return _ENGINE_LETTER.get(s, "unknown")
    return _class_from_description(s)


def _vote(classes: pd.Series) -> str:
    """Collapse per-aircraft votes for one designator into one class.

    Jet only when at least 95% of the classified aircraft agree; if jets are
    a minority above 5% the designator is contested and stays unknown.
    """
    known = classes[classes != "unknown"]
    if len(known) == 0:
        return "unknown"
    counts = known.value_counts()
    jet_share = counts.get("jet", 0) / len(known)
    if jet_share >= 0.95:
        return "jet"
    if jet_share > 0.05:
        return "unknown"
    non_jet = counts.drop("jet", errors="ignore")
    return str(non_jet.idxmax())


def load_type_table(csv_path: str | Path, *, override: bool = False) -> None:
    """Extend the live table from a CSV.

    Two layouts are accepted:

    * the OpenSky aircraft database export (``aircraftDatabase.csv``), which
      has one row per airframe with ``typecode`` and ``icaoaircrafttype``
      ("L2J", "L1P", ...). Rows are grouped by designator and voted on;
    * a generic file with columns ``typecode,engine_type`` where
      ``engine_type`` is a class name, a letter J/T/P/E, or an "L2J"-style
      description.

    By default a loaded entry only fills designators the built-in table does
    not know; pass ``override=True`` to let the file replace curated entries.
    """
    path = Path(csv_path)
    header = pd.read_csv(path, nrows=0, dtype=str, keep_default_na=False)
    cols = {c.strip().strip("'\"").lower(): c for c in header.columns}
    if "typecode" not in cols:
        raise ValueError(f"{path}: no 'typecode' column (columns: {list(header.columns)})")
    if "icaoaircrafttype" in cols:
        value_col = cols["icaoaircrafttype"]
    elif "engine_type" in cols:
        value_col = cols["engine_type"]
    else:
        raise ValueError(
            f"{path}: expected an 'icaoaircrafttype' (OpenSky) or 'engine_type' column"
        )
    df = pd.read_csv(
        path,
        usecols=[cols["typecode"], value_col],
        dtype=str,
        keep_default_na=False,
    )
    code = df[cols["typecode"]].map(_norm)
    cls = df[value_col].map(_class_from_token)
    keep = code != ""
    voted = cls[keep].groupby(code[keep]).agg(_vote)
    for designator, engine in voted.items():
        if engine == "unknown":
            continue
        if override or designator not in _BUILTIN:
            _TABLE[str(designator)] = str(engine)
