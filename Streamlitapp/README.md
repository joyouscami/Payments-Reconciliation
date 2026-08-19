# Bank & M-PESA B2C Reconciliation — Front End

A simple web front end for the fuzzy-matching reconciliation logic from
`Bank_and_MPESA_B2C_Reconciliation_with_RapidFuzz.ipynb`. Built with
[Streamlit](https://streamlit.io) so it runs as a local web app — upload two
Excel files, click a button, download the results.

## What it does

Same pipeline as the original notebook, just wrapped in a UI:

1. Validates that both uploaded files have the required columns.
2. Parses the date columns (`Initiation Time`, `payment_timestamp`) — tries
   the strict `DD-MM-YYYY HH:MM:SS` format first, then falls back to a
   flexible parser, and reports/drops any rows it can't parse.
3. Normalizes narration text and builds a TF-IDF similarity matrix to
   shortlist the top 10 candidate bank transactions per payment.
4. Scores each candidate on:
   - Narration similarity (RapidFuzz `token_set_ratio`) — 40%
   - Exact amount match — 30%
   - Date proximity (within 5 minutes) — 20%
   - Customer ID mention in the bank narration — 10%
5. Runs greedy one-to-one matching, sorted by confidence score.
6. Classifies each match: **Auto-Reconciled** (≥85), **Review Recommended**
   (≥70), **Weak Match** (≥50), or **Unmatched** (<50).
7. Produces the same output files as the notebook — detailed matches,
   summary, unmatched payments, unmatched bank transactions, and run
   metadata — as downloadable Excel files (individually or as one ZIP).

## Required file formats

**Payments / M-PESA file** must contain these columns (exact names):
`payment_ref`, `amount`, `Initiation Time`, `narration`
(optional: `customer_id`, used to boost confidence when present)

**Bank statement file** must contain these columns (exact names):
`bank_txn_ref`, `amount`, `payment_timestamp`, `narration`

## Setup & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501` in your browser. Upload both
files, click **Run Reconciliation**, and download the results once it
finishes.

## Notes

- For very large files (the original notebook tested with ~196k payment
  records against ~3,800 bank transactions), the matching step can take a
  few minutes — a progress bar shows how far through it is.
- All matching weights and thresholds (85 / 70 / 50, and the 40/30/20/10
  scoring split) are unchanged from the original notebook. If you want to
  tune them, they're clearly labeled near the top of `run_reconciliation()`
  in `app.py`.
