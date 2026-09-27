"""Convert the raw Lending Club CSVs (accepted 2.26M rows, rejected ~27.6M rows) to typed Parquet with DuckDB.

DuckDB streams the files, so this runs in a few GB of RAM regardless of input size.
"""
import glob
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
DATA = HERE / "data"


def find(pattern: str) -> str:
    hits = sorted(glob.glob(str(DATA / "**" / pattern), recursive=True), key=len)
    if not hits:
        raise SystemExit(f"No file matching {pattern} under data/. Run ./download_data.sh first.")
    return hits[0]


def pct(col: str) -> str:
    return f"TRY_CAST(replace(trim({col}), '%', '') AS DOUBLE)"


def month(col: str) -> str:
    return f"TRY_CAST(strptime(trim({col}), '%b-%Y') AS DATE)"


EMP = """CASE WHEN {c} LIKE '10+%' THEN 10 WHEN {c} LIKE '<%' THEN 0
             ELSE TRY_CAST(regexp_extract({c}, '(\\d+)', 1) AS INT) END"""


def main():
    (DATA / "tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='4GB'; SET threads=4; SET temp_directory='{DATA / 'tmp'}'; SET preserve_insertion_order=false")

    accepted = find("accepted_2007_to_2018*.csv*")
    print("accepted:", accepted)
    con.execute(f"""
    COPY (
      SELECT
        TRY_CAST(id AS BIGINT)                       AS id,
        TRY_CAST(loan_amnt AS DOUBLE)                AS loan_amnt,
        TRY_CAST(funded_amnt AS DOUBLE)              AS funded_amnt,
        TRY_CAST(regexp_extract(term, '(\\d+)', 1) AS INT) AS term,
        {pct('int_rate')}                            AS int_rate,
        TRY_CAST(installment AS DOUBLE)              AS installment,
        grade, sub_grade,
        {EMP.format(c='emp_length')}                 AS emp_length,
        home_ownership,
        TRY_CAST(annual_inc AS DOUBLE)               AS annual_inc,
        verification_status,
        {month('issue_d')}                           AS issue_d,
        loan_status, purpose, addr_state,
        TRY_CAST(dti AS DOUBLE)                      AS dti,
        TRY_CAST(delinq_2yrs AS DOUBLE)              AS delinq_2yrs,
        {month('earliest_cr_line')}                  AS earliest_cr_line,
        TRY_CAST(fico_range_low AS DOUBLE)           AS fico,
        TRY_CAST(inq_last_6mths AS DOUBLE)           AS inq_last_6mths,
        TRY_CAST(open_acc AS DOUBLE)                 AS open_acc,
        TRY_CAST(pub_rec AS DOUBLE)                  AS pub_rec,
        TRY_CAST(revol_bal AS DOUBLE)                AS revol_bal,
        {pct('revol_util')}                          AS revol_util,
        TRY_CAST(total_acc AS DOUBLE)                AS total_acc,
        initial_list_status, application_type,
        TRY_CAST(mort_acc AS DOUBLE)                 AS mort_acc,
        TRY_CAST(pub_rec_bankruptcies AS DOUBLE)     AS pub_rec_bankruptcies,
        TRY_CAST(total_pymnt AS DOUBLE)              AS total_pymnt,
        TRY_CAST(total_rec_prncp AS DOUBLE)          AS total_rec_prncp,
        TRY_CAST(recoveries AS DOUBLE)               AS recoveries,
        {month('last_pymnt_d')}                      AS last_pymnt_d
      FROM read_csv('{accepted}', header=true, all_varchar=true, ignore_errors=true)
      WHERE TRY_CAST(id AS BIGINT) IS NOT NULL AND TRY_CAST(funded_amnt AS DOUBLE) > 0
    ) TO '{DATA / 'accepted.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    print(con.execute(f"SELECT count(*), min(issue_d), max(issue_d) FROM '{DATA / 'accepted.parquet'}'").fetchall())

    rejected = find("rejected_2007_to_2018*.csv*")
    print("rejected:", rejected)
    con.execute(f"""
    COPY (
      SELECT
        TRY_CAST("Amount Requested" AS DOUBLE)            AS amount,
        TRY_CAST("Application Date" AS DATE)              AS app_date,
        TRY_CAST("Risk_Score" AS DOUBLE)                  AS risk_score,
        {pct('"Debt-To-Income Ratio"')}                   AS dti,
        "State"                                           AS state,
        {EMP.format(c='"Employment Length"')}             AS emp_length
      FROM read_csv('{rejected}', header=true, all_varchar=true, ignore_errors=true)
    ) TO '{DATA / 'rejected.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    print(con.execute(f"SELECT count(*), min(app_date), max(app_date) FROM '{DATA / 'rejected.parquet'}'").fetchall())


if __name__ == "__main__":
    main()
