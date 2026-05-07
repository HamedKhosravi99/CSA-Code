"""
Merge results/ltt_grid_summary.json into paper_tables/_verified_numbers.json.

For each (benchmark, alpha) pair present in the LTT grid summary, write
(or overwrite) the LTT cell in _verified_numbers.json using the same
field schema the file uses for every other method:
    risk  ar  maxr  prec  cov  pv

cov convention: the existing file uses cov=0 for the (1-2) LTT cells it
already has even when AR > 0; we match that convention for the new
cells so downstream table / figure scripts behave identically. prec is
filled as 1 - risk when AR > 0 (otherwise 1.0).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "results/ltt_grid_summary.json")
VFILE = os.path.join(HERE, "paper_tables/_verified_numbers.json")
BACKUP = VFILE + ".pre_ltt_grid.bak"

with open(GRID) as f:
    G = json.load(f)
with open(VFILE) as f:
    V = json.load(f)

# Keep a backup of the pre-merge file the first time we run.
if not os.path.exists(BACKUP):
    with open(BACKUP, "w") as f:
        json.dump(V, f, indent=2)
    print(f"Backup: {BACKUP}")

merged, new = 0, 0
for bench, alpha_map in G.items():
    bd = V["combined_main_plus_new"].setdefault(bench, {})
    for a_str, r in alpha_map.items():
        a_f = float(a_str)
        a_key = f"{a_f:.2g}".rstrip("0").rstrip(".") if "." in f"{a_f:.2g}" else f"{a_f:.2g}"
        # _verified_numbers uses keys like "0.05", "0.1", "0.15", "0.2",
        # "0.25", "0.3" -- normalize to those.
        a_key = str(a_f) if a_f >= 0.1 else "0.05"
        if abs(a_f - 0.05) < 1e-9: a_key = "0.05"
        elif abs(a_f - 0.10) < 1e-9: a_key = "0.1"
        elif abs(a_f - 0.15) < 1e-9: a_key = "0.15"
        elif abs(a_f - 0.20) < 1e-9: a_key = "0.2"
        elif abs(a_f - 0.25) < 1e-9: a_key = "0.25"
        elif abs(a_f - 0.30) < 1e-9: a_key = "0.3"
        cell = bd.setdefault(a_key, {})
        risk = float(r["final_risk_mean"])
        ar = float(r["final_ar_mean"])
        maxr = float(r["max_risk_mean"])
        prec = (1.0 - risk) if ar > 0 else 1.0
        pv = r["pathwise_violation_rate"]
        was_present = "LTT" in cell
        cell["LTT"] = {
            "risk": risk,
            "ar":   ar,
            "maxr": maxr,
            "prec": prec,
            "cov":  0.0,
            "pv":   pv,
        }
        if was_present:
            merged += 1
        else:
            new += 1
        print(f"  {bench:10s} a={a_key:4s}  Risk={risk:5.3f} AR={ar:5.3f} "
              f"PathV={pv}  ({'OVERWRITE' if was_present else 'NEW'})")

with open(VFILE, "w") as f:
    json.dump(V, f, indent=2)
print(f"\nWrote {VFILE}")
print(f"  new cells: {new}")
print(f"  overwritten cells: {merged}")
