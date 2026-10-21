
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from taxonomy.embed import ART

NONE = "None of these"
ABSTAIN = "Needs review"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; stays sensible at small n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- sample
def sample(args) -> None:
    out = Path(args.out)
    sheet = out / "to_label.csv"
    if sheet.exists() and not args.force:
        labelled = pd.read_csv(sheet, encoding="utf-8-sig")["label"].notna().sum()
        raise SystemExit(f"{sheet} already exists ({labelled} rows labelled). "
                         f"Pass --force to overwrite it.")

    con = duckdb.connect(args.db, read_only=True)
    if "conversation_topics" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
        raise SystemExit(f"no conversation_topics in {args.db}: run `predict` on it first")
    df = con.execute("""
        SELECT c.conversation_id, c.user_id, c.first_user_msg,
               p.category AS pred_primary, p.score AS pred_score, p.source AS pred_source,
               a.labels AS pred_labels
        FROM conversations c
        JOIN conversation_topics p ON p.conversation_id = c.conversation_id AND p.is_primary
        JOIN (SELECT conversation_id, string_agg(category, '; ') AS labels
              FROM conversation_topics GROUP BY 1) a ON a.conversation_id = c.conversation_id
    """).df()
    con.close()
    df["key"] = df["first_user_msg"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    # Random: plain random conversations, duplicates and heavy users included,
    # because that is the traffic the classifier will actually see.
    rand = df.sample(n=min(args.random, len(df)), random_state=args.seed).assign(sample="random")

    # Stratified: one row per distinct message, at most M per user, then up to
    # N per predicted category. Messages already in the random sample are skipped.
    pool = (df[~df["key"].isin(rand["key"])]
            .sample(frac=1, random_state=args.seed + 1)
            .drop_duplicates("key")
            .groupby("user_id").head(args.max_per_user))
    strat = pool.groupby("pred_primary").head(args.per_category).assign(sample="stratified")

    # Weight = conversations the row stands for: stratum size / rows drawn from it.
    pop = df["pred_primary"].value_counts()
    drawn = strat["pred_primary"].value_counts()
    strat["weight"] = strat["pred_primary"].map(pop / drawn)
    rand["weight"] = len(df) / len(rand)

    g = (pd.concat([rand, strat])
         .sample(frac=1, random_state=args.seed + 2)  # mix samples so the order gives nothing away
         .reset_index(drop=True))
    g.insert(0, "gold_id", [f"G{i:04d}" for i in range(1, len(g) + 1)])
    g = g.rename(columns={"pred_primary": "stratum_pred"})

    out.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens non-Latin text correctly
    (g[["gold_id", "first_user_msg"]].rename(columns={"first_user_msg": "first_user_message"})
     .assign(label="", label_2="", notes="")
     .to_csv(sheet, index=False, encoding="utf-8-sig"))
    g[["gold_id", "conversation_id", "user_id", "sample", "stratum_pred", "weight"]].to_csv(
        out / "design.csv", index=False)
    (g[["gold_id", "stratum_pred", "pred_score", "pred_source", "pred_labels"]]
     .rename(columns={"stratum_pred": "pred_primary"})
     .to_csv(out / "predictions.csv", index=False))

    tax = json.loads((ART / "taxonomy.json").read_text())
    lines = [f"# Labels — taxonomy {tax.get('version', '?')}", "",
             "Label each row by the **first user message**: what did the user want done?", "",
             "- `label`: exactly one of the names below, copied verbatim.",
             f"- `{NONE}`: the message fits no category (too vague, empty, or a topic the taxonomy lacks).",
             "- `label_2`: optional, only if the message clearly asks for two things.",
             "- `notes`: anything borderline — write down the rule you applied, and reuse it.",
             "- Don't look at `predictions.csv` until you've finished.", "", "## Categories", ""]
    lines += [f"- **{c['name']}** — {c.get('description', '')}" for c in tax["categories"]]
    lines += [f"- **{NONE}** — fits none of the above."]
    (out / "categories.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {sheet}: {len(g)} rows ({len(rand)} random + {len(strat)} stratified)")
    print("\nstratified rows per predicted category (weight = conversations each row stands for):")
    print(pd.DataFrame({"rows": drawn, "population": pop.loc[drawn.index],
                        "weight": (pop / drawn).loc[drawn.index].round(1)}).to_string())


# ---------------------------------------------------------------- evaluate
def evaluate(args) -> None:
    gold = Path(args.gold)
    labels = pd.read_csv(gold / "to_label.csv", encoding="utf-8-sig", dtype=str)
    design = pd.read_csv(gold / "design.csv")
    if args.db:  # the current labels in a predicted database, e.g. after changing the classifier
        con = duckdb.connect(args.db, read_only=True)
        preds = con.execute("SELECT conversation_id, category AS pred_primary FROM conversation_topics "
                            "WHERE is_primary").df().merge(design[["gold_id", "conversation_id"]])
        con.close()
    else:
        preds = pd.read_csv(args.predictions or gold / "predictions.csv", dtype={"pred_primary": str})
    tax = json.loads((ART / "taxonomy.json").read_text())
    allowed = {c["name"] for c in tax["categories"]} | {NONE}

    m = labels.merge(design, on="gold_id").merge(preds[["gold_id", "pred_primary"]], on="gold_id")
    for col in ("label", "label_2"):
        m[col] = m[col].fillna("").str.strip()
    m = m[m["label"] != ""]
    bad = m[~m["label"].isin(allowed)]
    if len(bad):
        print(f"⚠ {len(bad)} labels are not a category name and are ignored:")
        print(bad[["gold_id", "label"]].to_string(index=False), "\n")
        m = m[m["label"].isin(allowed)]
    if m.empty:
        raise SystemExit("no valid labels yet")

    # Abstaining ("Needs review") is right exactly when the gold label is "None of these".
    pred = m["pred_primary"].replace(ABSTAIN, NONE)
    m["correct"] = pred == m["label"]
    m["lenient"] = m["correct"] | (pred == m["label_2"])
    m["covered"] = m["pred_primary"] != ABSTAIN

    print(f"labelled: {len(m)} rows ({(m['sample'] == 'random').sum()} random, "
          f"{(m['sample'] == 'stratified').sum()} stratified)\n")

    r = m[m["sample"] == "random"]
    overall = {"source": str(args.db or args.predictions or gold / "predictions.csv"),
               "taxonomy_version": tax.get("version"), "labelled": len(m)}
    if len(r):
        print("== Overall (random sample) ==")
        for name, col in (("accuracy", "correct"), ("accuracy, label_2 also counts", "lenient")):
            k = int(r[col].sum()); lo, hi = wilson(k, len(r))
            print(f"  {name:<32} {k / len(r):6.1%}  (95% CI {lo:.0%}–{hi:.0%}, n={len(r)})")
            overall[col] = {"value": k / len(r), "ci95": [lo, hi], "n": len(r)}
        cov = r[r["covered"]]
        print(f"  {'coverage (not Needs review)':<32} {r['covered'].mean():6.1%}")
        overall["coverage"] = float(r["covered"].mean())
        if len(cov):
            print(f"  {'accuracy when it answers':<32} {cov['correct'].mean():6.1%}  (n={len(cov)})")
            overall["accuracy_when_answering"] = float(cov["correct"].mean())
        print()
        (gold / "summary.json").write_text(json.dumps(overall, indent=2))

    s = m[m["sample"] == "stratified"]
    if len(s):
        print("== Per category (stratified sample, weighted) ==")
        rows = []
        for c in sorted(allowed | {ABSTAIN}):
            target = NONE if c == ABSTAIN else c
            is_pred = s["pred_primary"] == c
            is_gold = s["label"] == target
            hit = is_pred & is_gold
            if c == NONE:
                continue  # covered by the Needs review row
            n_pred = int(is_pred.sum())
            lo, hi = wilson(int(hit.sum()), n_pred)
            # Weighted by the sampling design: rows were drawn per stratum of the
            # classifier that built the sheet, so another classifier's "predicted c"
            # mixes strata of different weight. For that classifier both agree.
            w = s["weight"]
            recall = (w[hit].sum() / w[is_gold].sum()) if is_gold.any() else np.nan
            rows.append({"category": c, "n_predicted": n_pred,
                         "precision": w[hit].sum() / w[is_pred].sum() if n_pred else np.nan,
                         "precision_95ci": f"{lo:.0%}–{hi:.0%}" if n_pred else "",
                         "recall_weighted": recall, "n_gold": int(is_gold.sum())})
        per = pd.DataFrame(rows).sort_values("n_predicted", ascending=False)
        print(per.to_string(index=False, float_format=lambda v: f"{v:.0%}"))
        per.to_csv(gold / "per_category.csv", index=False)

        nr = s[s["pred_primary"] == ABSTAIN]
        if len(nr):
            print(f"\n== What 'Needs review' actually is (n={len(nr)}) ==")
            print(nr["label"].value_counts().to_string())

        conf = pd.crosstab(s["label"], s["pred_primary"], rownames=["gold"], colnames=["predicted"])
        conf.to_csv(gold / "confusion.csv")
        print(f"\nwrote {gold / 'per_category.csv'} and {gold / 'confusion.csv'}")

    if args.second:
        b = pd.read_csv(args.second, encoding="utf-8-sig", dtype=str)[["gold_id", "label"]]
        both = m[["gold_id", "label"]].merge(b.dropna(), on="gold_id", suffixes=("_a", "_b"))
        both = both[both["label_b"].str.strip().isin(allowed)]
        if len(both):
            from sklearn.metrics import cohen_kappa_score
            agree = (both["label_a"] == both["label_b"].str.strip()).mean()
            kappa = cohen_kappa_score(both["label_a"], both["label_b"].str.strip())
            print(f"\n== Labeller agreement (n={len(both)}) ==\n  agreement {agree:.1%} · Cohen's kappa {kappa:.2f}")
