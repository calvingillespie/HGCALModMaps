#!/usr/bin/env python3
"""Generate a cassette/train JSON (e.g. fnal_CEH21D1.json) from the HGCAL modmap
(geometry_sipmontile_hgcal_*.txt, comma separated).

Usage:
  python make_cassette_json.py MODMAP.txt --cassette-type 21D --serial 1
  python make_cassette_json.py MODMAP.txt --cassette-type 21D --emp-channels emp_CEH21D1.txt

Conventions
  * Rows are selected by the `cassette_type` column (e.g. 21D).  Name = CEH<cassette_type><serial>
    (CEE for CE-E planes, i.e. plane < 27).  <serial> is just a serial number (--serial, default 1).
  * Modules with the same `MB` are on the same train.
      - BH train  (TM-* modules, isSiPM==1): TL1..TLn in MB order.
      - LD train  (ML-* modules, isSiPM==0): LD1..LDn in MB order.
  * position: increases moving away from the engine.  The isEngine module is closest to the
    engine (position 1 on its wagon).  Ordering uses the p,q columns relative to the engine's p,q.
    Cassettes A/C use q as the primary coordinate, B/D use p: modules sharing the engine's primary
    coordinate come first (by |d other|), then the rest (by |d primary|, then |d other|).
  * LD wagon 1 -> orientation "e", wagon 0 -> "w".  Modules are listed e1, e2, ..., w1, w2, ...
  * serial = CSV typecode with dashes removed (TM-G5FM -> TMG5FM).
  * BH "type" = 2 chars after "TM-", passed through TYPE_FIX (B1->B11, B2->B12; add others there).
    LD "type" = "full" if the type letter is F, otherwise "semi_or_half".
  * elinks are bit masks written as hex strings without 0x:
    n = engine_data_fibres (daq) / engine_trig_fibres (tpg) -> (1<<n)-1  (1->"1", 2->"3", 3->"7").
  * mapping fibre counts = max engine_data_fibres / engine_trig_fibres over the train's modules.
  * emp_channels come from --emp-channels (simple text file, see parse_emp_map); trains missing
    from it keep null.  TODO: the fibre labelling scheme is NOT derived: fibres are "TBD_*"
    unless supplied with --channels  (JSON: {"CEH21D1_TL1": {"daq":
    {"emp_channels": [48], "fibers": ["25_1_1_D_1"]}, "tpg": {...}}, ...}).
"""
import argparse, json, os
import numpy as np
import pandas as pd

# typecode letters+digit -> real tile type (not necessarily true for other tiles; extend as needed)
TYPE_FIX = {"B1": "B11", "B2": "B12"}


def mask_hex(n):
    return format((1 << int(n)) - 1, "x")


def fibre_block(n_d, n_t, tid):
    return {
        "daq": {"emp_channels": [None] * n_d,
                "fibers": [f"TBD_{tid}_D_{i+1}" for i in range(n_d)]},
        "tpg": {"emp_channels": [None] * n_t,
                "fibers": [f"TBD_{tid}_T_{i+1}" for i in range(n_t)]},
    }


def order_from_engine(g, primary):
    """Sort a group by distance from the isEngine module in the (p, q) plane.
    `primary` is the coordinate that takes priority ("p" or "q"): modules sharing the engine's
    primary coordinate come first (by |d other|), then the rest (by |d primary|, then |d other|)."""
    other = "q" if primary == "p" else "p"
    e = g[g.isEngine]
    r = e.iloc[0] if len(e) else g.sort_values(["p", "q"]).iloc[0]
    g = g.assign(_same=(g[primary] != r[primary]).astype(int),
                 _d1=(g[primary] - r[primary]).abs(), _d2=(g[other] - r[other]).abs())
    return g.sort_values(["_same", "_d1", "_d2", "v", "u"])


def groups_by_mb(df):
    """One group per MB, ordered by MB number."""
    return [g for _, g in sorted(df.groupby("MB"))]


def build_bh(cas, name, overrides, primary):
    trains = []
    for n, g in enumerate(groups_by_mb(cas[cas.isSiPM == 1]), start=1):
        tid = f"{name}_TL{n}"
        g = order_from_engine(g, primary)
        mods = [{"serial": r.typecode.replace("-", ""), "position": i,
                 "type": TYPE_FIX.get(r.typecode[3:5], r.typecode[3:5])}
                for i, r in enumerate(g.itertuples(), start=1)]
        mapping = overrides.get(tid) or fibre_block(
            int(g.engine_data_fibres.max()), int(g.engine_trig_fibres.max()), tid)
        trains.append({"train_id": tid, "type": "bh", "mapping": mapping, "modules": mods})
    return trains


def build_ld(cas, name, overrides, primary):
    trains = []
    for n, g in enumerate(groups_by_mb(cas[cas.isSiPM == 0]), start=1):
        tid = f"{name}_LD{n}"
        g = order_from_engine(g, primary)
        mods = []
        for wagon, orient in ((1, "e"), (0, "w")):
            w = g[g.wagon == wagon]  # already in engine order
            for pos, r in enumerate(w.itertuples(), start=1):
                mods.append({
                    "serial": r.typecode.replace("-", ""), "orientation": orient, "position": pos,
                    "type": "full" if r.typecode[3] == "F" else "semi_or_half",
                    "elinks": {"daq": mask_hex(r.engine_data_fibres),
                               "tpg": mask_hex(r.engine_trig_fibres)},
                })
        mapping = overrides.get(tid) or fibre_block(
            int(g.engine_data_fibres.max()), int(g.engine_trig_fibres.max()), tid)
        trains.append({"train_id": tid, "type": "ld", "mapping": mapping, "modules": mods})
    return trains


def parse_emp_map(path):
    """Parse the simple emp-channel map:

        CEH21D1_LD1:
            daq: 52
            trg: 84, 85        # 'tpg' is accepted too; '#' starts a comment
    Returns {train_id: {"daq": [..], "tpg": [..]}}."""
    out, cur = {}, None
    for n, raw in enumerate(open(path), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        key, _, val = line.strip().partition(":")
        if not line[0].isspace():                      # train header
            if val.strip():
                raise SystemExit(f"{path}:{n}: expected 'TRAIN_ID:' alone on the line")
            cur = out.setdefault(key.strip(), {})
        else:
            if cur is None:
                raise SystemExit(f"{path}:{n}: channel line before any train id")
            k = {"daq": "daq", "trg": "tpg", "tpg": "tpg", "trig": "tpg"}.get(key.strip().lower())
            if k is None:
                raise SystemExit(f"{path}:{n}: unknown key '{key.strip()}' (use daq / trg)")
            try:
                cur[k] = [int(x) for x in val.replace(",", " ").split()]
            except ValueError:
                raise SystemExit(f"{path}:{n}: channels must be integers: '{val.strip()}'")
    return out


def apply_emp_map(trains, emp):
    ids = {t["train_id"] for t in trains}
    for tid in emp:
        if tid not in ids:
            raise SystemExit(f"emp map has unknown train id {tid} (known: {sorted(ids)})")
    for t in trains:
        e = emp.get(t["train_id"])
        if e is None:
            print(f"WARNING: no emp channels given for {t['train_id']}")
            continue
        for k, chans in e.items():
            m = t["mapping"][k]
            if len(chans) != len(m["fibers"]):
                print(f"WARNING: {t['train_id']} {k}: {len(chans)} channels given but the modmap "
                      f"implies {len(m['fibers'])} fibres")
            m["emp_channels"] = chans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modmap")
    ap.add_argument("--cassette-type", required=True, help="e.g. 21D, or 'all' (with --plane)")
    ap.add_argument("--plane", type=int, help="only needed if cassette_type is not unique")
    ap.add_argument("--serial", type=int, default=1, help="cassette serial number (digit after the letter)")
    ap.add_argument("--prefix", default="fnal")
    ap.add_argument("--custom-data", default="auto-generated from modmap")
    ap.add_argument("--emp-channels", help="emp channel map file (see parse_emp_map)")
    ap.add_argument("--channels", help="JSON with emp_channels/fibers overrides per train_id")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    df = pd.read_csv(a.modmap)
    if a.plane is not None:
        df = df[df.plane == a.plane]
    types = sorted(df.cassette_type.unique()) if a.cassette_type == "all" else [a.cassette_type]
    overrides = json.load(open(a.channels)) if a.channels else {}
    emp = parse_emp_map(a.emp_channels) if a.emp_channels else {}
    os.makedirs(a.outdir, exist_ok=True)
    for ct in types:
        sub = df[df.cassette_type == ct]
        if sub.empty:
            raise SystemExit(f"cassette_type {ct} not found")
        if sub.plane.nunique() > 1:
            raise SystemExit(f"cassette_type {ct} spans planes {sorted(sub.plane.unique())}; use --plane")
        det = "CEH" if sub.plane.iloc[0] >= 27 else "CEE"
        name = f"{det}{ct}{a.serial}"
        primary = "q" if ct[-1] in "AC" else "p"   # A/C: q first, B/D: p first
        trains = build_bh(sub, name, overrides, primary) + build_ld(sub, name, overrides, primary)
        if emp:
            apply_emp_map(trains, emp)
        out = {"metadata": {"custom_data": a.custom_data}, "trains": trains}
        path = os.path.join(a.outdir, f"{a.prefix}_{name}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        print("wrote", path)


if __name__ == "__main__":
    main()
