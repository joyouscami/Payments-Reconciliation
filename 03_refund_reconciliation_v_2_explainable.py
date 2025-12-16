# 03_refund_reconciliation_v2_explainable.ipynb
# Explainable refund reconciliation using confidence scoring

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
data_dir = Path('C:/Payments Reconciliation')
out_dir = data_dir / 'outputs'
out_dir.mkdir(exist_ok=True)

payments = pd.read_excel(data_dir / 'payments_system.xlsx', parse_dates=['payment_timestamp'])
refunds = pd.read_excel(data_dir / 'refunds.xlsx', parse_dates=['refund_timestamp'])

# -----------------------------
# 2. Normalization utilities
# -----------------------------
def normalize_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

payments['norm_narration'] = payments['narration'].apply(normalize_text)
refunds['norm_narration'] = refunds['narration'].apply(normalize_text)

# -----------------------------
# 3. TF-IDF candidate generation
# -----------------------------
vectorizer = TfidfVectorizer(min_df=2)
all_text = pd.concat([
    payments['norm_narration'],
    refunds['norm_narration']
])

vectorizer.fit(all_text)

payment_vecs = vectorizer.transform(payments['norm_narration'])
refund_vecs = vectorizer.transform(refunds['norm_narration'])

similarity_matrix = cosine_similarity(refund_vecs, payment_vecs)

# -----------------------------
# 4. Explainable matching logic
# -----------------------------
results = []

for i, refund_row in refunds.iterrows():
    sim_scores = similarity_matrix[i]
    top_candidates = np.argsort(sim_scores)[-10:][::-1]

    for j in top_candidates:
        pay_row = payments.iloc[j]

        # Time to refund
        time_diff_days = (refund_row['refund_timestamp'] - pay_row['payment_timestamp']).days
        time_score = max(0, 1 - abs(time_diff_days) / 30)

        # Amount relationship
        refund_ratio = refund_row['refund_amount'] / pay_row['amount'] if pay_row['amount'] > 0 else 0
        amt_score = 1 if refund_ratio <= 1 else max(0, 1 - (refund_ratio - 1))

        # Narration similarity
        narr_score = fuzz.token_set_ratio(
            refund_row['norm_narration'],
            pay_row['norm_narration']
        ) / 100

        # Customer inference
        cust_score = 1 if str(pay_row['customer_id']) in refund_row['norm_narration'] else 0

        # Confidence score
        confidence = (
            0.45 * narr_score +
            0.25 * amt_score +
            0.2 * time_score +
            0.1 * cust_score
        ) * 100

        results.append({
            'refund_ref': refund_row['refund_ref'],
            'payment_ref': pay_row['payment_ref'],
            'confidence_score': round(confidence, 2),
            'narration_score': round(narr_score * 100, 2),
            'amount_score': round(amt_score * 100, 2),
            'timing_score': round(time_score * 100, 2),
            'customer_score': cust_score * 100,
            'refund_ratio': round(refund_ratio, 2),
            'days_to_refund': time_diff_days
        })

matches = pd.DataFrame(results)

# -----------------------------
# 5. Best match per refund
# -----------------------------
best_matches = matches.sort_values('confidence_score', ascending=False)
best_matches = best_matches.groupby('refund_ref').head(1).reset_index(drop=True)

# -----------------------------
# 6. Regulatory classification
# -----------------------------
def classify(score):
    if score >= 85:
        return 'Valid Refund'
    elif score >= 70:
        return 'Review Required'
    elif score >= 50:
        return 'High Risk Refund'
    else:
        return 'Unlinked Refund'

best_matches['refund_status'] = best_matches['confidence_score'].apply(classify)

# -----------------------------
# 7. Explainability flags
# -----------------------------
best_matches['explainability_note'] = np.select(
    [
        best_matches['refund_ratio'] > 1,
        best_matches['days_to_refund'] > 30,
        best_matches['narration_score'] < 50
    ],
    [
        'Refund exceeds original amount',
        'Unusually late refund',
        'Weak narration similarity'
    ],
    default='Within expected operational bounds'
)

# -----------------------------
# 8. Outputs for audit & regulators
# -----------------------------
best_matches.to_excel(out_dir / 'refund_reconciliation_detailed.xlsx', index=False)

summary = best_matches.groupby('refund_status').agg(
    refund_count=('refund_ref', 'count'),
    avg_confidence=('confidence_score', 'mean'),
    total_refund_amount=('refund_ratio', 'sum')
).reset_index()

summary.to_excel(out_dir / 'refund_reconciliation_summary.xlsx', index=False)

best_matches
