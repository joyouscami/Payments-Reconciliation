# 03_refund_reconciliation_v2_improved.ipynb
# Explainable refund reconciliation with production-ready improvements

import pandas as pd
import numpy as np
import re
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import sys

# -----------------------------
# 1. Setup with Error Handling
# -----------------------------
def setup_and_load_data():
    """Load refund and payment data with comprehensive error handling"""
    try:
        # Use relative path instead of hardcoded Windows path
        data_dir = Path('./Payments_Refunds')
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = data_dir / 'outputs' / f'refund_recon_{timestamp}'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directory created: {out_dir}")
        
        # Load payments
        try:
            payments_path = data_dir / 'payments_system.xlsx'
            if not payments_path.exists():
                raise FileNotFoundError(f"Payments file not found: {payments_path}")
            payments = pd.read_excel(payments_path, parse_dates=['payment_timestamp'])
            print(f"✓ Loaded {len(payments)} payment records")
        except Exception as e:
            print(f"✗ Error loading payments: {e}")
            raise
        
        # Load refunds
        try:
            refunds_path = data_dir / 'refunds.xlsx'
            if not refunds_path.exists():
                raise FileNotFoundError(f"Refunds file not found: {refunds_path}")
            refunds = pd.read_excel(refunds_path, parse_dates=['refund_timestamp'])
            print(f"✓ Loaded {len(refunds)} refund records")
        except Exception as e:
            print(f"✗ Error loading refunds: {e}")
            raise
        
        # Validate required columns
        required_payment_cols = ['payment_ref', 'amount', 'payment_timestamp', 'narration']
        required_refund_cols = ['refund_ref', 'refund_amount', 'refund_timestamp', 'narration']
        
        missing_payment_cols = [col for col in required_payment_cols if col not in payments.columns]
        missing_refund_cols = [col for col in required_refund_cols if col not in refunds.columns]
        
        if missing_payment_cols:
            raise ValueError(f"Missing required payment columns: {missing_payment_cols}")
        if missing_refund_cols:
            raise ValueError(f"Missing required refund columns: {missing_refund_cols}")
        
        # Validate refund_amount exists and handle if it's called 'amount'
        if 'refund_amount' not in refunds.columns and 'amount' in refunds.columns:
            refunds['refund_amount'] = refunds['amount']
            print("⚠ 'refund_amount' not found, using 'amount' column")
        
        print("✓ All required columns present")
        
        return payments, refunds, out_dir
        
    except Exception as e:
        print(f"✗ Fatal error during setup: {e}")
        sys.exit(1)

payments, refunds, out_dir = setup_and_load_data()

# -----------------------------
# 2. Text Normalization with Error Handling
# -----------------------------
def normalize_text(text):
    """Normalize text with proper null handling"""
    try:
        if pd.isna(text):
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"⚠ Warning: Error normalizing text '{text}': {e}")
        return ''

payments['norm_narration'] = payments['narration'].apply(normalize_text)
refunds['norm_narration'] = refunds['narration'].apply(normalize_text)

print("✓ Text normalization complete")

# -----------------------------
# 3. TF-IDF Candidate Generation with Error Handling
# -----------------------------
try:
    vectorizer = TfidfVectorizer(min_df=1)  # Changed from min_df=2 for small datasets
    all_text = pd.concat([
        payments['norm_narration'],
        refunds['norm_narration']
    ])
    
    # Check for empty text
    if all_text.empty or all_text.str.strip().eq('').all():
        raise ValueError("All narration fields are empty or invalid")
    
    vectorizer.fit(all_text)
    payment_vecs = vectorizer.transform(payments['norm_narration'])
    refund_vecs = vectorizer.transform(refunds['norm_narration'])
    similarity_matrix = cosine_similarity(refund_vecs, payment_vecs)
    
    print("✓ TF-IDF similarity matrix computed")
    
except Exception as e:
    print(f"✗ Error in candidate generation: {e}")
    sys.exit(1)

# -----------------------------
# 4. Safe Scoring Functions
# -----------------------------
def safe_time_score(refund_date, payment_date, max_days=30):
    """
    Calculate time score with proper handling of edge cases
    Refunds should come AFTER payments
    """
    try:
        time_diff_days = (refund_date - payment_date).days
        
        # Negative means refund before payment (suspicious)
        if time_diff_days < 0:
            return 0.0  # Red flag - refund before payment
        
        # Score decreases as refund gets further from payment
        score = max(0, 1 - abs(time_diff_days) / max_days)
        return score
        
    except Exception as e:
        print(f"⚠ Warning: Error calculating time score: {e}")
        return 0.0

def safe_amount_ratio(refund_amount, payment_amount):
    """
    Calculate amount ratio score with proper handling of edge cases
    Refund should not exceed original payment
    """
    try:
        # Handle zero or negative payment amounts
        if payment_amount <= 0:
            return 0.0
        
        # Handle negative refund amounts (should be positive)
        if refund_amount < 0:
            print(f"⚠ Warning: Negative refund amount detected: {refund_amount}")
            refund_amount = abs(refund_amount)
        
        # Calculate ratio
        refund_ratio = refund_amount / payment_amount
        
        # Perfect score if refund <= payment
        if refund_ratio <= 1:
            return 1.0
        
        # Penalize refunds exceeding payment (potential fraud)
        # Rapid decline for over-refunds
        score = max(0, 1 - (refund_ratio - 1) * 2)
        return score
        
    except Exception as e:
        print(f"⚠ Warning: Error calculating amount ratio: {e}")
        return 0.0

# -----------------------------
# 5. Explainable Matching Logic
# -----------------------------
results = []
used_payment_refs = set()  # Track matched payments for one-to-one constraint

print("Starting refund matching process...")

for i, refund_row in refunds.iterrows():
    try:
        # Candidate shortlist
        sim_scores = similarity_matrix[i]
        top_candidates = np.argsort(sim_scores)[-10:][::-1]

        for j in top_candidates:
            pay_row = payments.iloc[j]
            
            # Skip if payment already matched (one-to-one constraint)
            # Note: In refunds, we might want one-to-many (one payment, multiple partial refunds)
            # For now, we'll collect all matches and handle duplicates later
            
            # Time to refund score
            try:
                time_diff_days = (refund_row['refund_timestamp'] - pay_row['payment_timestamp']).days
                time_score = safe_time_score(refund_row['refund_timestamp'], pay_row['payment_timestamp'])
            except Exception as e:
                print(f"⚠ Warning: Date calculation error for refund {refund_row['refund_ref']}: {e}")
                time_diff_days = None
                time_score = 0.0

            # Amount relationship score
            refund_ratio = safe_amount_ratio(refund_row['refund_amount'], pay_row['amount'])
            amt_score = refund_ratio
            actual_ratio = refund_row['refund_amount'] / pay_row['amount'] if pay_row['amount'] > 0 else 0

            # Narration similarity score
            try:
                narr_score = fuzz.token_set_ratio(
                    refund_row['norm_narration'],
                    pay_row['norm_narration']
                ) / 100
            except Exception as e:
                print(f"⚠ Warning: Narration matching error: {e}")
                narr_score = 0.0

            # Customer inference
            try:
                cust_score = 1 if str(pay_row.get('customer_id', '')) in refund_row['norm_narration'] else 0
            except Exception as e:
                cust_score = 0

            # Weighted confidence score (adjusted for refunds)
            confidence = (
                0.45 * narr_score +
                0.25 * amt_score +
                0.20 * time_score +
                0.10 * cust_score
            ) * 100

            results.append({
                'refund_ref': refund_row['refund_ref'],
                'refund_amount': refund_row['refund_amount'],
                'refund_date': refund_row['refund_timestamp'],
                'payment_ref': pay_row['payment_ref'],
                'payment_amount': pay_row['amount'],
                'payment_date': pay_row['payment_timestamp'],
                'confidence_score': round(confidence, 2),
                'narration_score': round(narr_score * 100, 2),
                'amount_score': round(amt_score * 100, 2),
                'timing_score': round(time_score * 100, 2),
                'customer_score': cust_score * 100,
                'refund_ratio': round(actual_ratio, 2),
                'days_to_refund': time_diff_days
            })
            
    except Exception as e:
        print(f"⚠ Warning: Error processing refund {refund_row.get('refund_ref', 'unknown')}: {e}")
        continue

if not results:
    print("✗ No matches found!")
    sys.exit(1)

matches = pd.DataFrame(results)
print(f"✓ Generated {len(matches)} potential refund matches")

# -----------------------------
# 6. Best Match Selection (Allowing Partial Refunds)
# -----------------------------
# For refunds, we need special handling:
# - One payment can have multiple partial refunds
# - But one refund should match only one payment

# Sort by confidence and get best match per refund
best_matches = matches.sort_values('confidence_score', ascending=False)
best_matches = best_matches.groupby('refund_ref').head(1).reset_index(drop=True)

print(f"✓ Selected {len(best_matches)} best matches (1 payment per refund)")

# Identify payments with multiple refunds
payment_refund_counts = best_matches.groupby('payment_ref').size()
multi_refund_payments = payment_refund_counts[payment_refund_counts > 1]

if len(multi_refund_payments) > 0:
    print(f"⚠ Found {len(multi_refund_payments)} payments with multiple refunds")

# -----------------------------
# 7. Regulatory Classification
# -----------------------------
def classify_refund(score):
    if score >= 85:
        return 'Valid Refund'
    elif score >= 70:
        return 'Review Required'
    elif score >= 50:
        return 'High Risk Refund'
    else:
        return 'Unlinked Refund'

best_matches['refund_status'] = best_matches['confidence_score'].apply(classify_refund)

# -----------------------------
# 8. Enhanced Explainability Flags
# -----------------------------
def generate_explainability_note(row):
    """Generate detailed explainability notes for audit purposes"""
    flags = []
    
    # Amount-based flags
    if row['refund_ratio'] > 1:
        flags.append(f"⚠ ALERT: Refund exceeds payment by {(row['refund_ratio'] - 1) * 100:.1f}%")
    elif row['refund_ratio'] > 0.95:
        flags.append("Full refund detected")
    elif row['refund_ratio'] < 0.5:
        flags.append("Partial refund (<50%)")
    
    # Timing-based flags
    if row['days_to_refund'] is not None:
        if row['days_to_refund'] < 0:
            flags.append("⚠ CRITICAL: Refund dated BEFORE original payment")
        elif row['days_to_refund'] > 90:
            flags.append(f"⚠ Late refund ({row['days_to_refund']} days)")
        elif row['days_to_refund'] > 30:
            flags.append(f"Delayed refund ({row['days_to_refund']} days)")
    
    # Narration-based flags
    if row['narration_score'] < 50:
        flags.append("⚠ Weak narration similarity - verify manually")
    
    # Score-based flags
    if row['confidence_score'] < 70:
        flags.append("Low confidence match")
    
    if not flags:
        return "Within expected operational bounds"
    
    return " | ".join(flags)

best_matches['explainability_note'] = best_matches.apply(generate_explainability_note, axis=1)

# Add risk flag
best_matches['risk_flag'] = best_matches['explainability_note'].str.contains('⚠').fillna(False)

# -----------------------------
# 9. Create Unmatched Reports
# -----------------------------
# Unmatched refunds (potential fraud or system issues)
unmatched_refunds = refunds[~refunds['refund_ref'].isin(best_matches['refund_ref'])].copy()
unmatched_refunds['reason'] = 'No matching payment found - possible orphan refund'
print(f"⚠ {len(unmatched_refunds)} unmatched refunds (INVESTIGATE)")

# Payments with no refunds (informational only)
payments_with_refunds = payments[payments['payment_ref'].isin(best_matches['payment_ref'])].copy()
print(f"ℹ {len(payments_with_refunds)} payments have associated refunds")

# -----------------------------
# 10. Audit-Friendly Outputs with Timestamps
# -----------------------------
try:
    # Main refund reconciliation results
    best_matches_sorted = best_matches.sort_values('confidence_score', ascending=False)
    best_matches_sorted.to_excel(out_dir / 'refund_reconciliation_detailed.xlsx', index=False)
    print(f"✓ Saved: refund_reconciliation_detailed.xlsx")
    
    # High-risk refunds for immediate review
    high_risk = best_matches[best_matches['risk_flag'] == True].copy()
    if not high_risk.empty:
        high_risk.to_excel(out_dir / 'high_risk_refunds_REVIEW_IMMEDIATELY.xlsx', index=False)
        print(f"✓ Saved: high_risk_refunds_REVIEW_IMMEDIATELY.xlsx ({len(high_risk)} records)")
    
    # Summary statistics by status
    summary = best_matches.groupby('refund_status').agg(
        refund_count=('refund_ref', 'count'),
        total_refund_amount=('refund_amount', 'sum'),
        avg_confidence=('confidence_score', 'mean'),
        avg_refund_ratio=('refund_ratio', 'mean')
    ).reset_index()
    summary.to_excel(out_dir / 'refund_reconciliation_summary.xlsx', index=False)
    print(f"✓ Saved: refund_reconciliation_summary.xlsx")
    
    # Unmatched refunds report (CRITICAL for fraud detection)
    if not unmatched_refunds.empty:
        unmatched_refunds.to_excel(out_dir / 'unmatched_refunds_INVESTIGATE.xlsx', index=False)
        print(f"✓ Saved: unmatched_refunds_INVESTIGATE.xlsx ({len(unmatched_refunds)} records)")
    
    # Multi-refund analysis (payments with multiple refunds)
    if len(multi_refund_payments) > 0:
        multi_refund_detail = best_matches[
            best_matches['payment_ref'].isin(multi_refund_payments.index)
        ].sort_values(['payment_ref', 'refund_date'])
        multi_refund_detail.to_excel(out_dir / 'payments_with_multiple_refunds.xlsx', index=False)
        print(f"✓ Saved: payments_with_multiple_refunds.xlsx ({len(multi_refund_payments)} payments)")
    
    # Reconciliation metadata
    metadata = pd.DataFrame([{
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_refunds': len(refunds),
        'matched_refunds': len(best_matches),
        'unmatched_refunds': len(unmatched_refunds),
        'valid_refunds': len(best_matches[best_matches['refund_status'] == 'Valid Refund']),
        'review_required': len(best_matches[best_matches['refund_status'] == 'Review Required']),
        'high_risk': len(best_matches[best_matches['refund_status'] == 'High Risk Refund']),
        'refunds_with_flags': len(high_risk),
        'payments_with_refunds': len(payments_with_refunds),
        'payments_with_multi_refunds': len(multi_refund_payments),
        'match_rate': f"{len(best_matches) / len(refunds) * 100:.1f}%",
        'total_refund_value': best_matches['refund_amount'].sum(),
        'avg_days_to_refund': best_matches['days_to_refund'].mean()
    }])
    metadata.to_excel(out_dir / 'refund_reconciliation_metadata.xlsx', index=False)
    print(f"✓ Saved: refund_reconciliation_metadata.xlsx")
    
    # Console summary
    print(f"\n{'='*60}")
    print(f"REFUND RECONCILIATION COMPLETE")
    print(f"{'='*60}")
    print(f"Matched: {len(best_matches)} / {len(refunds)} refunds ({len(best_matches)/len(refunds)*100:.1f}%)")
    print(f"Valid refunds: {len(best_matches[best_matches['refund_status'] == 'Valid Refund'])}")
    print(f"Review required: {len(best_matches[best_matches['refund_status'] == 'Review Required'])}")
    print(f"High risk: {len(best_matches[best_matches['refund_status'] == 'High Risk Refund'])}")
    print(f"Unmatched refunds: {len(unmatched_refunds)} ⚠ INVESTIGATE")
    print(f"Refunds with risk flags: {len(high_risk)} ⚠ REVIEW IMMEDIATELY")
    print(f"Payments with multiple refunds: {len(multi_refund_payments)}")
    print(f"\nAll outputs saved to: {out_dir}")
    
    if len(unmatched_refunds) > 0:
        print(f"\n⚠ ACTION REQUIRED: {len(unmatched_refunds)} orphan refunds need investigation")
    if len(high_risk) > 0:
        print(f"⚠ ACTION REQUIRED: {len(high_risk)} high-risk refunds flagged")
    
except Exception as e:
    print(f"✗ Error saving outputs: {e}")
    sys.exit(1)

# Display results
best_matches_sorted.head(20)