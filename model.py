"""Default and return models, out of time: train on loans issued 2007-2014, pick from 36-month loans issued in 2015.

-> results/model.json
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from analyze import ANN_RET, BAD, DONE, MATURED

SEED = 42
HERE = Path(__file__).parent

NUMERIC = ["loan_amnt", "term", "int_rate", "installment", "emp_length", "annual_inc", "dti", "delinq_2yrs", "fico",
           "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
           "pub_rec_bankruptcies", "credit_age_months", "loan_to_income", "payment_to_income", "sub_grade_rank"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type"]
LC_PRICING = ["int_rate", "sub_grade_rank", "installment"]


def load() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4")
    df = con.execute(f"""
      SELECT *, ({BAD})::INT AS bad, {ANN_RET} AS ann_ret,
             date_diff('month', earliest_cr_line, issue_d) AS credit_age_months,
             loan_amnt / nullif(annual_inc, 0) AS loan_to_income,
             installment * 12 / nullif(annual_inc, 0) AS payment_to_income,
             (ascii(substr(sub_grade, 1, 1)) - 65) * 5 + TRY_CAST(substr(sub_grade, 2, 1) AS INT) AS sub_grade_rank
      FROM '{HERE / 'data' / 'accepted.parquet'}'
      WHERE {MATURED} AND {DONE}""").df()
    df["ann_ret"] = df["ann_ret"].clip(-1, 0.5)
    return df


def pipe(estimator, numeric, categorical):
    prep = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), categorical)])
    estimator.set_params(categorical_features=[False] * len(numeric) + [True] * len(categorical))
    return Pipeline([("prep", prep), ("m", estimator)])


def clf():
    return HistGradientBoostingClassifier(learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          random_state=SEED)


def reg():
    return HistGradientBoostingRegressor(learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         random_state=SEED)


def pooled(df):
    """Annualised return of a dollar-weighted portfolio of 36-month loans, held to the end."""
    return float((df["total_pymnt"].sum() / df["funded_amnt"].sum()) ** (1 / 3) - 1)


def summary(label, sel):
    return {"strategy": label, "loans": int(len(sel)), "default_rate": round(float(sel["bad"].mean()), 4),
            "avg_int_rate": round(float(sel["int_rate"].mean()), 2), "pooled_ann_return": round(pooled(sel), 4)}


def main():
    df = load()
    print("matured, resolved loans:", len(df))
    train = df[df["issue_d"] < pd.Timestamp("2015-01-01")].copy()
    test = df[(df["issue_d"].dt.year == 2015) & (df["term"] == 36)].copy()
    print("train", len(train), "test", len(test))
    y_tr, y_te = train["bad"].to_numpy(), test["bad"].to_numpy()

    cols = NUMERIC + CATEGORICAL
    no_lc = [c for c in NUMERIC if c not in LC_PRICING]
    auc = {"Lending Club sub-grade alone": round(float(roc_auc_score(y_te, test["sub_grade_rank"])), 4)}
    pd_nolc = pipe(clf(), no_lc, CATEGORICAL).fit(train[no_lc + CATEGORICAL], y_tr)
    auc["Model without LC grade or rate"] = round(float(roc_auc_score(y_te, pd_nolc.predict_proba(test[no_lc + CATEGORICAL])[:, 1])), 4)
    pd_model = pipe(clf(), NUMERIC, CATEGORICAL).fit(train[cols], y_tr)
    test["pd"] = pd_model.predict_proba(test[cols])[:, 1]
    auc["Model with LC grade and rate"] = round(float(roc_auc_score(y_te, test["pd"])), 4)
    print(auc)

    ret_model = pipe(reg(), NUMERIC, CATEGORICAL).fit(train[cols], train["ann_ret"])
    test["pred_ret"] = ret_model.predict(test[cols])

    strategies = [summary("Buy every 2015 36-month loan", test)]
    for g in "ABCDEFG":
        sel = test[test["grade"] == g]
        if len(sel):
            strategies.append(summary(f"Grade {g} only", sel))
    for share in (0.1, 0.2, 0.3):
        k = int(share * len(test))
        strategies.append(summary(f"Highest interest rate, top {int(share * 100)}%", test.nlargest(k, "int_rate")))
        strategies.append(summary(f"Lowest predicted default, top {int(share * 100)}%", test.nsmallest(k, "pd")))
        strategies.append(summary(f"Highest predicted return, top {int(share * 100)}%", test.nlargest(k, "pred_ret")))

    within_grade = []
    for g in "ABCDEFG":
        sel = test[test["grade"] == g]
        if len(sel) < 500:
            continue
        keep = sel[sel["pd"] <= sel["pd"].quantile(0.5)]
        within_grade.append({"grade": g, "all_return": round(pooled(sel), 4), "kept_half_return": round(pooled(keep), 4),
                             "all_default": round(float(sel["bad"].mean()), 4),
                             "kept_half_default": round(float(keep["bad"].mean()), 4)})

    test["dec"] = pd.qcut(test["pd"], 10, labels=False, duplicates="drop")
    calib = test.groupby("dec").agg(pred=("pd", "mean"), obs=("bad", "mean"), n=("bad", "size")).reset_index()

    sample = test.sample(40000, random_state=SEED)
    imp = permutation_importance(pd_model, sample[cols], sample["bad"], scoring="roc_auc", n_repeats=3,
                                 random_state=SEED, n_jobs=2)
    importances = sorted([{"feature": f, "auc_drop": round(float(v), 4)} for f, v in zip(cols, imp.importances_mean)],
                         key=lambda d: -d["auc_drop"])[:12]

    out = {"train_loans": int(len(train)), "test_loans": int(len(test)),
           "train_bad_rate": round(float(y_tr.mean()), 4), "test_bad_rate": round(float(y_te.mean()), 4),
           "auc": auc, "strategies": strategies, "within_grade": within_grade,
           "calibration": [{"pred": round(r.pred, 4), "obs": round(r.obs, 4), "n": int(r.n)} for r in calib.itertuples()],
           "importances": importances}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "model.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("auc", "strategies", "within_grade")}, indent=1))


if __name__ == "__main__":
    main()
