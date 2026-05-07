"""
Merge tight-alpha data into paper_tables/_verified_numbers.json for
GSM8K and ARC so the riskandar / phase-budget figures can show the
regime below the base-model error rate.

Two data sources:

  (A) Existing per-benchmark tight-alpha JSONs that cover 6 methods
      (CSA, Always-Act, Fixed-Threshold, Naive-Tuning, ACI, SAOCP):
          results/gsm8k/gsm8k_alpha0.01.json
          results/gsm8k/gsm8k_alpha0.03.json
          results/gsm8k/gsm8k_alpha0.07.json        (alpha 0.075)
          results/arc/arc_alpha0.07.json            (alpha 0.075)
          results/arc/arc_alpha0.05.json            (alpha 0.05, already
                                                    fully covered in main
                                                    grid; we skip unless
                                                    methods are missing)

  (B) New tight-alpha run produced by run_tight_alpha_gsm8k_arc.py:
          results/tight_alpha_gsm8k_arc.json
      Covers LTT, CRC, NEX-Conf, Mohri-Conf at:
          gsm8k alpha in {0.01, 0.03, 0.075}
          arc   alpha in {0.05, 0.075}

Both sources are normalized to the schema used by _verified_numbers.json:
    { "risk": r, "ar": a, "maxr": mr, "prec": 1-r (or 1.0),
      "cov": 0.0, "pv": "pv/N" }
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
VFILE = os.path.join(HERE, "paper_tables/_verified_numbers.json")
TIGHT_JSON = os.path.join(HERE, "results/tight_alpha_gsm8k_arc.json")
BACKUP = VFILE + ".pre_tight_alpha.bak"

EXISTING_FILES = {
    "gsm8k": [
        ("0.01",  "results/gsm8k/gsm8k_alpha0.01.json"),
        ("0.03",  "results/gsm8k/gsm8k_alpha0.03.json"),
        ("0.075", "results/gsm8k/gsm8k_alpha0.07.json"),
    ],
    "arc": [
        ("0.075", "results/arc/arc_alpha0.07.json"),
        # arc alpha 0.05 is already fully covered in main grid.
    ],
}


def to_schema(r, a, mr, pv_str):
    """Map a method's raw numbers to _verified_numbers.json schema."""
    prec = (1.0 - r) if a > 0 else 1.0
    return {
        "risk": float(r),
        "ar":   float(a),
        "maxr": float(mr),
        "prec": float(prec),
        "cov":  0.0,
        "pv":   pv_str,
    }


def load_existing_six_method_json(path):
    """Return dict method -> (risk, ar, maxr, pv) or empty if missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for m, v in d.get("methods", {}).items():
        out[m] = to_schema(
            v.get("final_risk_mean", 0.0),
            v.get("final_ar_mean", 0.0),
            v.get("max_risk_mean", v.get("final_risk_mean", 0.0)),
            v.get("pathwise_violation_rate", f"{v.get('pathwise_violations', 0)}/{v.get('n_reps', 10)}"),
        )
    return out


def main():
    with open(VFILE) as f:
        V = json.load(f)
    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            json.dump(V, f, indent=2)
        print(f"Backup: {BACKUP}")

    total_added = total_skipped = 0

    # ---- Source A: existing 6-method JSONs -----------------------------
    for bench, files in EXISTING_FILES.items():
        bd = V["combined_main_plus_new"].setdefault(bench, {})
        for a_key, path in files:
            cells = load_existing_six_method_json(path)
            if not cells:
                print(f"[SKIP] {path}: not found")
                continue
            cell = bd.setdefault(a_key, {})
            for m, sch in cells.items():
                if m in cell:
                    total_skipped += 1
                    continue
                cell[m] = sch
                total_added += 1
                print(f"  [A] {bench} a={a_key} {m:16s} risk={sch['risk']:.3f} ar={sch['ar']:.3f} pv={sch['pv']}")

    # ---- Source B: new tight-alpha run ---------------------------------
    if os.path.exists(TIGHT_JSON):
        with open(TIGHT_JSON) as f:
            G = json.load(f)
        for bench, amap in G.items():
            bd = V["combined_main_plus_new"].setdefault(bench, {})
            for a_key_raw, methods in amap.items():
                a_key = a_key_raw
                cell = bd.setdefault(a_key, {})
                for m, v in methods.items():
                    sch = to_schema(
                        v["final_risk_mean"],
                        v["final_ar_mean"],
                        v["max_risk_mean"],
                        v["pathwise_violation_rate"],
                    )
                    cell[m] = sch
                    total_added += 1
                    print(f"  [B] {bench} a={a_key} {m:16s} risk={sch['risk']:.3f} ar={sch['ar']:.3f} pv={sch['pv']}")
    else:
        print(f"[WARN] {TIGHT_JSON} not found -- run run_tight_alpha_gsm8k_arc.py first")

    with open(VFILE, "w") as f:
        json.dump(V, f, indent=2)
    print(f"\nWrote {VFILE}")
    print(f"  cells added (new):        {total_added}")
    print(f"  cells skipped (existed):  {total_skipped}")


if __name__ == "__main__":
    main()
