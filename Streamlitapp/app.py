"""
Bank & M-PESA B2C Reconciliation Tool
--------------------------------------
A simple front end around the fuzzy-matching reconciliation logic
(TF-IDF candidate shortlist + RapidFuzz narration scoring + exact
amount matching + date-proximity scoring) that was originally built
as a Jupyter notebook.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Bank & M-PESA Reconciliation",
    page_icon="🧾",
    layout="wide",
)

REQUIRED_PAYMENT_COLS = ["payment_ref", "amount", "Initiation Time", "narration"]
REQUIRED_BANK_COLS = ["bank_txn_ref", "amount", "payment_timestamp", "narration"]
OPTIONAL_PAYMENT_COLS = ["customer_id"]

st.title("🧾 Bank & M-PESA B2C Reconciliation")
st.write(
    "Upload your **M-PESA B2C payments** file and your **bank statement** file, "
    "and this tool will run the same fuzzy-matching reconciliation logic as the "
    "original script — TF-IDF candidate shortlisting, RapidFuzz narration "
    "similarity, exact amount matching, and a 5-minute date-proximity window — "
    "and give you downloadable Excel results."
)

with st.expander("📋 Required file formats", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Payments / M-PESA file must contain:**")
        for c in REQUIRED_PAYMENT_COLS:
            st.markdown(f"- `{c}`")
        st.markdown("Optional (improves matching):")
        for c in OPTIONAL_PAYMENT_COLS:
            st.markdown(f"- `{c}`")
    with col2:
        st.markdown("**Bank statement file must contain:**")
        for c in REQUIRED_BANK_COLS:
            st.markdown(f"- `{c}`")
    st.caption(
        "Column names must match exactly (case-sensitive). Date columns can be "
        "in `DD-MM-YYYY HH:MM:SS` format or any format pandas can parse — the "
        "app will try the strict format first and fall back automatically."
    )

# -----------------------------------------------------------------------
# Helpers (same logic as the original notebook)
# -----------------------------------------------------------------------


def normalize_text(text):
    """Normalize text with proper null handling."""
    try:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


def exact_amount_score(bank_amount, payment_amount):
    """Exact amount match only. Handles negative/zero amounts safely."""
    try:
        if abs(bank_amount) == abs(payment_amount):
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def parse_datetime_column(series: pd.Series, label: str, warnings: list):
    """Try the strict DD-MM-YYYY HH:MM:SS format first, fall back to a
    flexible parse, and record any rows that fail to parse."""
    strict = pd.to_datetime(series, format="%d-%m-%Y %H:%M:%S", errors="coerce")
    if strict.isna().sum() == 0:
        return strict

    flexible = pd.to_datetime(series, errors="coerce", dayfirst=True)
    n_failed = flexible.isna().sum()
    if n_failed > 0:
        warnings.append(
            f"⚠ {n_failed} row(s) in '{label}' had a date that could not be "
            f"parsed and will be excluded from matching."
        )
    return flexible


def load_and_validate(payments_file, bank_file):
    """Load the two uploaded files and validate required columns."""
    errors = []

    try:
        payments = pd.read_excel(payments_file)
    except Exception as e:
        errors.append(f"Could not read the payments file: {e}")
        payments = None

    try:
        bank = pd.read_excel(bank_file)
    except Exception as e:
        errors.append(f"Could not read the bank statement file: {e}")
        bank = None

    if payments is not None:
        missing = [c for c in REQUIRED_PAYMENT_COLS if c not in payments.columns]
        if missing:
            errors.append(f"Payments file is missing required column(s): {missing}")

    if bank is not None:
        missing = [c for c in REQUIRED_BANK_COLS if c not in bank.columns]
        if missing:
            errors.append(f"Bank file is missing required column(s): {missing}")

    return payments, bank, errors


def run_reconciliation(payments: pd.DataFrame, bank: pd.DataFrame, progress_callback=None):
    """Run the full reconciliation pipeline. Mirrors the original notebook's
    logic exactly (weights, thresholds, candidate shortlist size, etc.)."""

    warnings = []

    payments = payments.copy()
    bank = bank.copy()

    # ---- Parse dates -------------------------------------------------
    payments["Initiation Time"] = parse_datetime_column(
        payments["Initiation Time"], "Initiation Time (payments)", warnings
    )
    bank["payment_timestamp"] = parse_datetime_column(
        bank["payment_timestamp"], "payment_timestamp (bank)", warnings
    )

    before_p, before_b = len(payments), len(bank)
    payments = payments.dropna(subset=["Initiation Time"]).reset_index(drop=True)
    bank = bank.dropna(subset=["payment_timestamp"]).reset_index(drop=True)
    if len(payments) < before_p:
        warnings.append(
            f"⚠ Dropped {before_p - len(payments)} payment row(s) with unparseable dates."
        )
    if len(bank) < before_b:
        warnings.append(
            f"⚠ Dropped {before_b - len(bank)} bank row(s) with unparseable dates."
        )

    # ---- Text normalization -------------------------------------------
    payments["norm_narration"] = payments["narration"].apply(normalize_text)
    bank["norm_narration"] = bank["narration"].apply(normalize_text)

    # ---- TF-IDF candidate generation -----------------------------------
    vectorizer = TfidfVectorizer(min_df=1)
    all_text = pd.concat([payments["norm_narration"], bank["norm_narration"]])
    if all_text.empty or all_text.str.strip().eq("").all():
        raise ValueError("All narration fields are empty or invalid — cannot match.")

    vectorizer.fit(all_text)
    payment_vecs = vectorizer.transform(payments["norm_narration"])
    bank_vecs = vectorizer.transform(bank["norm_narration"])
    similarity_matrix = cosine_similarity(payment_vecs, bank_vecs)

    # ---- Matching loop ---------------------------------------------
    results = []
    used_bank_refs = set()  # kept for parity with original script (not used to filter here)
    total = len(payments)
    has_customer_id = "customer_id" in payments.columns

    for i, pay_row in payments.iterrows():
        try:
            sim_scores = similarity_matrix[i]
            top_candidates = np.argsort(sim_scores)[-10:][::-1]

            for j in top_candidates:
                bank_row = bank.iloc[j]

                # Date proximity score
                try:
                    time_diff = (
                        abs(bank_row["payment_timestamp"] - pay_row["Initiation Time"]).total_seconds()
                        / 60
                    )
                    date_score = 1.0 if time_diff <= 5 else 0.0
                except Exception:
                    date_score = 0.0

                # Amount score
                amt_score = exact_amount_score(bank_row["amount"], pay_row["amount"])

                # Narration similarity
                try:
                    narr_score = (
                        fuzz.token_set_ratio(pay_row["norm_narration"], bank_row["norm_narration"]) / 100
                    )
                except Exception:
                    narr_score = 0.0

                # Customer inference
                try:
                    cust_score = (
                        1
                        if has_customer_id and str(pay_row.get("customer_id", "")) in bank_row["norm_narration"]
                        else 0
                    )
                except Exception:
                    cust_score = 0

                confidence = (0.4 * narr_score + 0.3 * amt_score + 0.2 * date_score + 0.1 * cust_score) * 100

                results.append(
                    {
                        "payment_ref": pay_row["payment_ref"],
                        "payment_amount": pay_row["amount"],
                        "payment_date": pay_row["Initiation Time"],
                        "payment_narration": pay_row["narration"],
                        "bank_txn_ref": bank_row["bank_txn_ref"],
                        "bank_amount": bank_row["amount"],
                        "bank_date": bank_row["payment_timestamp"],
                        "bank_narration": bank_row["narration"],
                        "confidence_score": round(confidence, 2),
                        "narration_score": round(narr_score * 100, 2),
                        "amount_score": round(amt_score * 100, 2),
                        "date_score": round(date_score * 100, 2),
                        "customer_score": cust_score * 100,
                    }
                )
        except Exception as e:
            warnings.append(f"⚠ Error processing payment {pay_row.get('payment_ref', 'unknown')}: {e}")
            continue

        if progress_callback and (i % 500 == 0 or i == total - 1):
            progress_callback(min((i + 1) / total, 1.0))

    if not results:
        raise ValueError("No candidate matches were generated — check your data.")

    matches = pd.DataFrame(results)

    # ---- One-to-one greedy matching -----------------------------------
    matches_sorted = matches.sort_values("confidence_score", ascending=False).reset_index(drop=True)
    best_matches = []
    matched_payments, matched_bank_txns = set(), set()

    for _, match in matches_sorted.iterrows():
        pay_ref, bank_ref = match["payment_ref"], match["bank_txn_ref"]
        if pay_ref not in matched_payments and bank_ref not in matched_bank_txns:
            best_matches.append(match)
            matched_payments.add(pay_ref)
            matched_bank_txns.add(bank_ref)

    best_matches = pd.DataFrame(best_matches)

    # ---- Classification -------------------------------------------
    def classify(score):
        if score >= 85:
            return "Auto-Reconciled"
        elif score >= 70:
            return "Review Recommended"
        elif score >= 50:
            return "Weak Match"
        return "Unmatched"

    best_matches["recon_status"] = best_matches["confidence_score"].apply(classify)

    # ---- Unmatched reports -------------------------------------------
    unmatched_payments = payments[~payments["payment_ref"].isin(best_matches["payment_ref"])].copy()
    unmatched_payments["reason"] = "No suitable bank transaction found"

    unmatched_bank = bank[~bank["bank_txn_ref"].isin(best_matches["bank_txn_ref"])].copy()
    unmatched_bank["reason"] = "No matching payment record found"

    # ---- Summary -------------------------------------------------
    if not best_matches.empty:
        summary = (
            best_matches.groupby("recon_status")
            .agg(
                transaction_count=("payment_ref", "count"),
                total_amount=("payment_amount", "sum"),
                avg_confidence=("confidence_score", "mean"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            columns=["recon_status", "transaction_count", "total_amount", "avg_confidence"]
        )

    metadata = pd.DataFrame(
        [
            {
                "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_payments": len(payments),
                "total_bank_txns": len(bank),
                "matched_count": len(best_matches),
                "unmatched_payments": len(unmatched_payments),
                "unmatched_bank_txns": len(unmatched_bank),
                "auto_reconciled": int((best_matches["recon_status"] == "Auto-Reconciled").sum())
                if not best_matches.empty
                else 0,
                "needs_review": int((best_matches["recon_status"] == "Review Recommended").sum())
                if not best_matches.empty
                else 0,
                "match_rate": f"{len(best_matches) / len(payments) * 100:.1f}%" if len(payments) else "0%",
            }
        ]
    )

    return {
        "best_matches": best_matches,
        "summary": summary,
        "unmatched_payments": unmatched_payments,
        "unmatched_bank": unmatched_bank,
        "metadata": metadata,
        "warnings": warnings,
    }


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def build_zip(outputs: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("reconciliation_results_detailed.xlsx", df_to_excel_bytes(outputs["best_matches"]))
        zf.writestr("reconciliation_summary.xlsx", df_to_excel_bytes(outputs["summary"]))
        if not outputs["unmatched_payments"].empty:
            zf.writestr("unmatched_payments.xlsx", df_to_excel_bytes(outputs["unmatched_payments"]))
        if not outputs["unmatched_bank"].empty:
            zf.writestr("unmatched_bank_transactions.xlsx", df_to_excel_bytes(outputs["unmatched_bank"]))
        zf.writestr("reconciliation_metadata.xlsx", df_to_excel_bytes(outputs["metadata"]))
    return buf.getvalue()


# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------

col_a, col_b = st.columns(2)
with col_a:
    payments_file = st.file_uploader(
        "1️⃣ Payments / M-PESA B2C file (.xlsx)", type=["xlsx"], key="payments"
    )
with col_b:
    bank_file = st.file_uploader("2️⃣ Bank statement file (.xlsx)", type=["xlsx"], key="bank")

run_clicked = st.button("🔄 Run Reconciliation", type="primary", disabled=not (payments_file and bank_file))

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_clicked:
    payments, bank, errors = load_and_validate(payments_file, bank_file)

    if errors:
        for e in errors:
            st.error(e)
    else:
        st.info(f"Loaded {len(payments):,} payment records and {len(bank):,} bank transactions.")
        progress_bar = st.progress(0.0, text="Matching payments to bank transactions…")

        def update_progress(frac):
            progress_bar.progress(frac, text=f"Matching payments to bank transactions… {frac*100:.0f}%")

        try:
            with st.spinner("Running reconciliation — this can take a while for large files…"):
                outputs = run_reconciliation(payments, bank, progress_callback=update_progress)
            progress_bar.progress(1.0, text="Done")
            st.session_state["results"] = outputs
            st.success("Reconciliation complete!")
        except Exception as e:
            st.error(f"Reconciliation failed: {e}")
            st.session_state["results"] = None

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------
outputs = st.session_state.get("results")

if outputs:
    if outputs["warnings"]:
        with st.expander(f"⚠ {len(outputs['warnings'])} warning(s) during processing"):
            for w in outputs["warnings"]:
                st.write(w)

    meta = outputs["metadata"].iloc[0]
    st.subheader("Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total payments", f"{meta['total_payments']:,}")
    m2.metric("Total bank txns", f"{meta['total_bank_txns']:,}")
    m3.metric("Matched", f"{meta['matched_count']:,}", meta["match_rate"])
    m4.metric("Auto-reconciled", f"{meta['auto_reconciled']:,}")
    m5.metric("Needs review", f"{meta['needs_review']:,}")

    st.dataframe(outputs["summary"], use_container_width=True)

    st.subheader("Matched transactions (preview)")
    st.dataframe(outputs["best_matches"].head(200), use_container_width=True)

    st.subheader("⬇️ Download results")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button(
        "Detailed matches",
        df_to_excel_bytes(outputs["best_matches"]),
        file_name="reconciliation_results_detailed.xlsx",
    )
    d2.download_button(
        "Summary",
        df_to_excel_bytes(outputs["summary"]),
        file_name="reconciliation_summary.xlsx",
    )
    d3.download_button(
        "Unmatched payments",
        df_to_excel_bytes(outputs["unmatched_payments"]),
        file_name="unmatched_payments.xlsx",
        disabled=outputs["unmatched_payments"].empty,
    )
    d4.download_button(
        "Unmatched bank txns",
        df_to_excel_bytes(outputs["unmatched_bank"]),
        file_name="unmatched_bank_transactions.xlsx",
        disabled=outputs["unmatched_bank"].empty,
    )
    d5.download_button(
        "Metadata",
        df_to_excel_bytes(outputs["metadata"]),
        file_name="reconciliation_metadata.xlsx",
    )

    st.download_button(
        "📦 Download all as ZIP",
        build_zip(outputs),
        file_name="reconciliation_outputs.zip",
        type="primary",
    )
else:
    st.caption("Upload both files and click **Run Reconciliation** to get started.")
