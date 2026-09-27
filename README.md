# Lending Club: which loans should an investor buy?

An end-to-end analysis of **2,260,668 funded loans ($34.0B) and 27,648,741 rejected applications**
from Lending Club, 2007–2018. The question is an investor's: does the extra interest on riskier loans
pay for the extra losses, and can a model trained on older loans pick better new ones than
Lending Club's own grades?

**Data:** [Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) on
Kaggle, about 3.5 GB of CSV. DuckDB streams it into Parquet and runs every aggregate out of core, so
the whole pipeline fits in 8 GB of RAM.

## Headline results

| Buying 2015 36-month loans (283,026 loans, never seen in training) | Realised return | Charge-off rate |
|---|---|---|
| Every loan | 2.14% a year | 14.9% |
| Grade A only | 2.25% | 5.4% |
| Highest interest rates, top 20% | 1.49% | 26.9% |
| Lowest predicted default, top 20% | 2.40% | 4.2% |
| **Highest predicted return, top 20%** | **2.91%** | 13.9% |
| **Highest predicted return, top 10%** | **3.16%** | 14.0% |

- **Chasing yield loses money.** Grades E, F and G charge 19–27% interest but their 2015 loans realised 0.7%, −0.6% and −1.8% a year.
- **The model adds about 0.8 points a year** over buying the whole book, with a default rate no higher than average.
- **Inside every grade, the safer half pays more.** Dropping the riskier half by predicted default lifts
  grade D from 1.61% to 2.29% and grade E from 0.72% to 1.69%.

![Strategies](figures/strategies.png)

## What the data shows

![Volume and approval](figures/volume_approval.png)

Volume grew from $0.02B in 2008 to $7.9B in 2018 while applications grew faster: the approval rate
peaked at 15.1% in 2013 and fell to 5.0% in 2018. Among 2016–2018 applications, approval falls from
about 11% at a 15–20% debt-to-income ratio to under 1% above 40%.

![Vintage heatmap](figures/vintage_heatmap.png)

Loss rates by grade were stable from 2009 to 2013, then rose for C–G loans issued in 2014 and 2015,
the years volume grew fastest. Grade F charged off 27% of 2013 loans and 42% of 2015 loans.

![Sub-grade economics](figures/subgrade_economics.png)

Interest rates rise in a straight line across sub-grades; losses rise faster. Realised return peaks
around B4 (3.1% a year) and falls away below grade D.

## Method

1. **Ingest** (`ingest.py`): DuckDB reads the gzipped CSVs as text, casts 34 useful columns (rates,
   dates such as `Dec-2015`, employment length) and writes ZSTD Parquet. 30M rows in a few minutes.
2. **Only count finished loans.** The snapshot ends in December 2018, so a loan's outcome is only final
   once its term has run: 36-month loans issued up to 2015 and 60-month loans up to 2013. That leaves
   676,302 resolved loans; counting still-running loans as "good" would understate losses badly.
3. **Analyse** (`analyze.py`): approval rates, grade mix, vintage curves, sub-grade economics, purpose and FICO bands, all in SQL.
4. **Model** (`model.py`): gradient boosting on what an investor sees at issuance only: loan terms,
   income, DTI, FICO, credit history, purpose, state, and Lending Club's grade and rate. Anything
   recorded after issuance (payments, recoveries, last FICO) is excluded. Trained on 393,276 loans
   issued 2007–2014; tested on 283,026 36-month loans issued in 2015.
   - A default classifier (AUC and importance) and a return regressor (for picking loans).
5. **Figures** (`figures.py`).

| Default model, 2015 test loans | AUC |
|---|---|
| Lending Club sub-grade alone | 0.679 |
| Model without Lending Club's grade or rate | 0.676 |
| Model with grade and rate | **0.696** |

Borrower data alone matches Lending Club's own grading; combining both is best. The model
under-predicts 2015 defaults slightly (29.2% predicted vs 32.4% observed in the riskiest decile),
consistent with the worse 2015 vintage above.

<p><img src="figures/importance.png" width="60%"></p>

## Definitions and limits
- **Realised return** = (total cash received ÷ amount funded)^(1/3) − 1 for 36-month loans, pooled
  dollar-weighted. It ignores reinvesting early repayments, so it understates true IRR for every
  strategy equally; the comparisons between strategies are the point.
- **Charge-off** includes `Charged Off`, `Default` and the older "does not meet the credit policy" charge-offs.
- Fees and taxes are not modelled. Past Lending Club performance says nothing about any current product.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_data.sh      # needs ~/.kaggle/kaggle.json
python ingest.py        # CSV -> Parquet
python analyze.py       # results/analysis.json
python model.py         # results/model.json
python figures.py       # figures/*.png
```
