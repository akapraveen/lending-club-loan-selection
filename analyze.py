"""Portfolio analytics over the full Lending Club book with DuckDB -> results/analysis.json."""
import json
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
A = HERE / "data" / "accepted.parquet"
R = HERE / "data" / "rejected.parquet"

# A loan's outcome is only final once its term has run: the snapshot ends in 2018Q4.
MATURED = "((term = 36 AND issue_d <= DATE '2015-12-01') OR (term = 60 AND issue_d <= DATE '2013-12-01'))"
BAD = "loan_status IN ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off')"
DONE = BAD.replace(")", ", 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid')")
ANN_RET = "power(total_pymnt / funded_amnt, 12.0 / term) - 1"


def rows(con, sql):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [{c: (round(v, 6) if isinstance(v, float) else (str(v) if hasattr(v, "isoformat") else v))
             for c, v in zip(cols, r)} for r in cur.fetchall()]


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4")
    con.execute(f"CREATE VIEW a AS SELECT * FROM '{A}'")
    con.execute(f"CREATE VIEW r AS SELECT * FROM '{R}'")
    con.execute(f"CREATE VIEW m AS SELECT *, ({BAD})::INT AS bad, {ANN_RET} AS ann_ret FROM a WHERE {MATURED} AND {DONE}")

    out = {}
    out["overview"] = rows(con, f"""
      SELECT (SELECT count(*) FROM a) AS accepted_loans, (SELECT count(*) FROM r) AS rejected_applications,
             (SELECT sum(funded_amnt) FROM a) AS funded_usd,
             (SELECT min(issue_d) FROM a) AS first_issue, (SELECT max(issue_d) FROM a) AS last_issue,
             (SELECT count(*) FROM m) AS matured_resolved_loans,
             (SELECT count(*) FROM a WHERE {MATURED}) AS matured_loans,
             (SELECT avg(bad) FROM m) AS matured_bad_rate""")[0]

    out["by_year"] = rows(con, """
      WITH acc AS (SELECT year(issue_d) y, count(*) n, sum(funded_amnt) usd, avg(int_rate) rate FROM a GROUP BY 1),
           rej AS (SELECT year(app_date) y, count(*) n FROM r WHERE app_date IS NOT NULL GROUP BY 1)
      SELECT acc.y AS year, acc.n AS accepted, rej.n AS rejected, acc.n / (acc.n + rej.n) AS approval_rate,
             acc.usd AS funded_usd, acc.rate AS avg_int_rate
      FROM acc JOIN rej USING (y) ORDER BY 1""")

    out["approval_by_dti"] = rows(con, """
      WITH x AS (
        SELECT least(floor(dti / 5) * 5, 50) AS band, 1 AS accepted FROM a WHERE issue_d >= DATE '2016-01-01' AND dti >= 0
        UNION ALL
        SELECT least(floor(dti / 5) * 5, 50), 0 FROM r WHERE app_date >= DATE '2016-01-01' AND dti >= 0)
      SELECT band AS dti_band, count(*) AS applications, avg(accepted) AS approval_rate FROM x GROUP BY 1 ORDER BY 1""")

    out["grade_mix_by_year"] = rows(con, """
      SELECT year(issue_d) AS year, grade, count(*) * 1.0 / sum(count(*)) OVER (PARTITION BY year(issue_d)) AS share
      FROM a GROUP BY 1, 2 ORDER BY 1, 2""")

    out["vintage"] = rows(con, """
      SELECT year(issue_d) AS year, grade, count(*) AS loans, avg(bad) AS bad_rate
      FROM m WHERE term = 36 GROUP BY 1, 2 HAVING count(*) >= 200 ORDER BY 1, 2""")

    out["sub_grade"] = rows(con, """
      SELECT sub_grade, count(*) AS loans, avg(int_rate) AS avg_int_rate, avg(bad) AS bad_rate,
             avg(ann_ret) AS avg_ann_return,
             power(sum(total_pymnt) / sum(funded_amnt), 1 / 3.0) - 1 AS pooled_ann_return,
             avg(CASE WHEN bad = 1 THEN (total_pymnt) / funded_amnt END) AS recovered_share_on_default
      FROM m WHERE term = 36 AND issue_d >= DATE '2010-01-01' GROUP BY 1 ORDER BY 1""")

    out["grade_term"] = rows(con, """
      SELECT grade, term, count(*) AS loans, avg(bad) AS bad_rate, avg(int_rate) AS avg_int_rate, avg(ann_ret) AS avg_ann_return
      FROM m WHERE issue_d >= DATE '2010-01-01' GROUP BY 1, 2 ORDER BY 1, 2""")

    out["purpose"] = rows(con, """
      SELECT purpose, count(*) AS loans, avg(bad) AS bad_rate, avg(int_rate) AS avg_int_rate
      FROM m WHERE term = 36 GROUP BY 1 HAVING count(*) >= 5000 ORDER BY bad_rate DESC""")

    out["fico_band"] = rows(con, """
      SELECT least(floor(fico / 20) * 20, 800) AS fico_band, count(*) AS loans, avg(bad) AS bad_rate, avg(int_rate) AS avg_int_rate
      FROM m WHERE term = 36 GROUP BY 1 HAVING count(*) >= 1000 ORDER BY 1""")

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["overview"], indent=1, default=str))


if __name__ == "__main__":
    main()
