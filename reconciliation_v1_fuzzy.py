# 02_reconciliation_v2_fuzzy.ipynb
# Real-world 3-way reconciliation using fuzzy matching and confidence scoring

import pandas as pd
import numpy as np
import re
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Setup
# -----------------------------
data_dir = Path('C:/Revenue Leakage')
out_dir = data_dir / 'outputs_v2'
out_dir.mkdir(exist_ok=True)

payments = pd.read_csv(data_dir / 'payments_system_v2.csv', parse_dates=['payment_timestamp'])
bank = pd.read_csv(data_dir / 'bank_statement_v2.csv', parse_dates=['bank_posting_timestamp'])
refunds = pd.read_csv(data_dir / 'refunds_v2.csv', parse_dates=['refund_timestamp'])

# -----------------------------
# 2. Text normalization utilities
# -----------------------------
def normalize_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

payments['norm_narration'] = payments['narration'].apply(normalize_text)
bank['norm_narration'] = bank['narration'].apply(normalize_text)

# -----------------------------
# 3. Candidate generation using TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(min_df=2)
all_text = pd.concat([
payments['norm_narration'],
    bank['norm_narration']
])

vectorizer.fit(all_text)

payment_vecs = vectorizer.transform(payments['norm_narration'])
bank_vecs = vectorizer.transform(bank['norm_narration'])

similarity_matrix = cosine_similarity(payment_vecs, bank_vecs)

# -----------------------------
# 4. Matching logic with confidence scoring
# -----------------------------
results = []

for i, pay_row in payments.iterrows():
    # candidate shortlist
    sim_scores = similarity_matrix[i]
    top_candidates = np.argsort(sim_scores)[-10:][::-1]

    for j in top_candidates:
        bank_row = bank.iloc[j]

        # Date proximity score
        date_diff_days = abs((bank_row['bank_posting_timestamp'] - pay_row['payment_timestamp']).days)
        date_score = max(0, 1 - date_diff_days / 7)

        # Amount proximity score
        amt_diff = abs(bank_row['net_amount'] - pay_row['amount'])
        amt_score = max(0, 1 - (amt_diff / pay_row['amount']))

        # Narration similarity score
        narr_score = fuzz.token_set_ratio(
            pay_row['norm_narration'],
            bank_row['norm_narration']
        ) / 100

        # Customer inference
        cust_score = 1 if str(pay_row['customer_id']) in bank_row['norm_narration'] else 0

        # Weighted confidence score
        confidence = (
            0.4 * narr_score +
            0.3 * amt_score +
            0.2 * date_score +
            0.1 * cust_score
        ) * 100

        results.append({
            'payment_ref': pay_row['payment_ref'],
            'bank_txn_ref': bank_row['bank_txn_ref'],
            'confidence_score': round(confidence, 2),
            'narration_score': round(narr_score * 100, 2),
            'amount_score': round(amt_score * 100, 2),
            'date_score': round(date_score * 100, 2),
            'customer_score': cust_score * 100
        })

matches = pd.DataFrame(results)

# -----------------------------
# 5. Select best match per payment
# -----------------------------
best_matches = matches.sort_values('confidence_score', ascending=False)
best_matches = best_matches.groupby('payment_ref').head(1).reset_index(drop=True)

# -----------------------------
# 6. Classification for audit
# -----------------------------
def classify(score):
    if score >= 85:
        return 'Auto-Reconciled'
    elif score >= 70:
        return 'Review Recommended'
    elif score >= 50:
        return 'Weak Match'
    else:
        return 'Unmatched'

best_matches['recon_status'] = best_matches['confidence_score'].apply(classify)

# -----------------------------
# 7. Audit-friendly outputs
# -----------------------------
best_matches.to_csv(out_dir / 'reconciliation_results_detailed.csv', index=False)

summary = best_matches.groupby('recon_status').agg(
    transaction_count=('payment_ref', 'count'),
    avg_confidence=('confidence_score', 'mean')
).reset_index()

summary.to_csv(out_dir / 'reconciliation_summary.csv', index=False)

best_matches
