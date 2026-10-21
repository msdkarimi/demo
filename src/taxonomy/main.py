import argparse
from taxonomy.embed import embed
from taxonomy.classify import classify


def add_label_args(c):
    c.add_argument("--min-sim", type=float, default=0.3, help="below this -> 'Needs review'")
    c.add_argument("--margin", type=float, default=0.05, help="multi-label closeness to best")
    c.add_argument("--max-labels", type=int, default=3)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--db", default="data/wildchat.duckdb")
    d.add_argument("--k", type=int, default=30, help="number of clusters")
    d.add_argument("--embedder", choices=["st", "tfidf"], default="st")
    d.add_argument("--llm", action="store_true")
    d.add_argument("--model", default="claude-sonnet-5-5")
    c = sub.add_parser("classify", help="label the conversations discover was fitted on")
    c.add_argument("--db", default="data/wildchat.duckdb")
    add_label_args(c)
    pr = sub.add_parser("predict", help="label any other conversations database (e.g. held-out)")
    pr.add_argument("--db", required=True)
    add_label_args(pr)
    gs = sub.add_parser("gold-sample", help="draw the labelling sheet from a predicted held-out db")
    gs.add_argument("--db", default="data/holdout/wildchat.duckdb")
    gs.add_argument("--out", default="gold")
    gs.add_argument("--random", type=int, default=100, help="plain random conversations")
    gs.add_argument("--per-category", type=int, default=20, help="stratified rows per predicted category")
    gs.add_argument("--max-per-user", type=int, default=2)
    gs.add_argument("--seed", type=int, default=0)
    gs.add_argument("--force", action="store_true", help="overwrite an existing to_label.csv")
    ge = sub.add_parser("gold-eval", help="score predictions against the gold labels")
    ge.add_argument("--gold", default="gold")
    ge.add_argument("--predictions", help="CSV with gold_id, pred_primary (default: gold/predictions.csv)")
    ge.add_argument("--db", help="score the current conversation_topics in this db instead, "
                                 "e.g. data/holdout/wildchat.duckdb after re-running predict")
    ge.add_argument("--second", help="a second labeller's to_label.csv, for agreement")
    args = p.parse_args()

    if args.cmd == "discover":
        embed(args)
    elif args.cmd == "classify":
        classify(args)
    elif args.cmd == "predict":
        from taxonomy.predict import predict_db
        predict_db(args)
    elif args.cmd == "gold-sample":
        from taxonomy.gold import sample
        sample(args)
    else:
        from taxonomy.gold import evaluate
        evaluate(args)


if __name__ == "__main__":
    main()
