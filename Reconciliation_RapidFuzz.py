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
def setup_directories_and_load_data():
    """Load data with comprehensive error handling"""
    try:
        # Use relative path instead of hardcoded Windows path
        data_dir = Path('./Payments Reconciliation')
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = data_dir / 'outputs' / timestamp
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
        
        # Load bank statement
        try:
            bank_path = data_dir / 'bank_statement.xlsx'
            if not bank_path.exists():
                raise FileNotFoundError(f"Bank statement not found: {bank_path}")
            bank = pd.read_excel(bank_path, parse_dates=['posting_date'])
            print(f"✓ Loaded {len(bank)} bank transactions")
        except Exception as e:
            print(f"✗ Error loading bank statement: {e}")
            raise
        
        # Load refunds (optional)
        try:
            refunds_path = data_dir / 'refunds.xlsx'
            if refunds_path.exists():
                refunds = pd.read_excel(refunds_path, parse_dates=['refund_timestamp'])
                print(f"✓ Loaded {len(refunds)} refund records")
            else:
                refunds = pd.DataFrame()
                print("⚠ No refunds file found, continuing without refunds")
        except Exception as e:
            print(f"⚠ Error loading refunds (non-critical): {e}")
            refunds = pd.DataFrame()
        
        # Validate required columns
        required_payment_cols = ['payment_ref', 'amount', 'payment_timestamp', 'narration']
        required_bank_cols = ['bank_txn_ref', 'amount', 'posting_date', 'narration']
        
        missing_payment_cols = [col for col in required_payment_cols if col not in payments.columns]
        missing_bank_cols = [col for col in required_bank_cols if col not in bank.columns]
        
        if missing_payment_cols:
            raise ValueError(f"Missing required payment columns: {missing_payment_cols}")
        if missing_bank_cols:
            raise ValueError(f"Missing required bank columns: {missing_bank_cols}")
        
        print("✓ All required columns present")
        
        return payments, bank, refunds, out_dir
        
    except Exception as e:
        print(f"✗ Fatal error during setup: {e}")
        sys.exit(1)

payments, bank, refunds, out_dir = setup_directories_and_load_data()

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
bank['norm_narration'] = bank['narration'].apply(normalize_text)

print("✓ Text normalization complete")

# -----------------------------
# 3. Candidate Generation with Error Handling
# -----------------------------
try:
    vectorizer = TfidfVectorizer(min_df=1)  # Changed from min_df=2 to handle small datasets
    all_text = pd.concat([
        payments['norm_narration'],
        bank['norm_narration']
    ])
    
    # Check for empty text
    if all_text.empty or all_text.str.strip().eq('').all():
        raise ValueError("All narration fields are empty or invalid")
    
    vectorizer.fit(all_text)
    payment_vecs = vectorizer.transform(payments['norm_narration'])
    bank_vecs = vectorizer.transform(bank['norm_narration'])
    similarity_matrix = cosine_similarity(payment_vecs, bank_vecs)
    
    print("✓ TF-IDF similarity matrix computed")
    
except Exception as e:
    print(f"✗ Error in candidate generation: {e}")
    sys.exit(1)

# -----------------------------
# 4. Matching Logic with Safe Amount Handling
# -----------------------------
def safe_amount_score(bank_amount, payment_amount, tolerance=0.01):
    """
    Calculate amount score with proper handling of edge cases
    
    Args:
        bank_amount: Amount from bank statement
        payment_amount: Amount from payment system
        tolerance: Acceptable difference percentage (default 1%)
    """
    try:
        # Handle zero or negative amounts
        if payment_amount == 0:
            return 1.0 if bank_amount == 0 else 0.0
        
        if payment_amount < 0 or bank_amount < 0:
            # For negative amounts (refunds), compare absolute values
            amt_diff = abs(abs(bank_amount) - abs(payment_amount))
            reference = abs(payment_amount)
        else:
            amt_diff = abs(bank_amount - payment_amount)
            reference = payment_amount
        
        # Calculate percentage difference
        pct_diff = amt_diff / reference if reference != 0 else 1.0
        
        # Score based on tolerance
        if pct_diff <= tolerance:
            return 1.0  # Within tolerance, perfect match
        else:
            score = max(0, 1 - pct_diff)
            return score
            
    except Exception as e:
        print(f"⚠ Warning: Error calculating amount score: {e}")
        return 0.0

results = []
used_bank_refs = set()  # Track used bank transactions for one-to-one matching

print("Starting matching process...")

for i, pay_row in payments.iterrows():
    try:
        # Candidate shortlist
        sim_scores = similarity_matrix[i]
        top_candidates = np.argsort(sim_scores)[-10:][::-1]

        for j in top_candidates:
            bank_row = bank.iloc[j]
            
            # Skip if bank transaction already matched (one-to-one constraint)
            if bank_row['bank_txn_ref'] in used_bank_refs:
                continue

            # Date proximity score
            try:
                date_diff_days = abs((bank_row['posting_date'] - pay_row['payment_timestamp']).days)
                date_score = max(0, 1 - date_diff_days / 7)
            except Exception as e:
                print(f"⚠ Warning: Date calculation error for payment {pay_row['payment_ref']}: {e}")
                date_score = 0.0

            # Amount proximity score with safe handling
            amt_score = safe_amount_score(bank_row['amount'], pay_row['amount'])

            # Narration similarity score
            try:
                narr_score = fuzz.token_set_ratio(
                    pay_row['norm_narration'],
                    bank_row['norm_narration']
                ) / 100
            except Exception as e:
                print(f"⚠ Warning: Narration matching error: {e}")
                narr_score = 0.0

            # Customer inference
            try:
                cust_score = 1 if str(pay_row.get('customer_id', '')) in bank_row['norm_narration'] else 0
            except Exception as e:
                cust_score = 0

            # Weighted confidence score
            confidence = (
                0.4 * narr_score +
                0.3 * amt_score +
                0.2 * date_score +
                0.1 * cust_score
            ) * 100

            results.append({
                'payment_ref': pay_row['payment_ref'],
                'payment_amount': pay_row['amount'],
                'payment_date': pay_row['payment_timestamp'],
                'bank_txn_ref': bank_row['bank_txn_ref'],
                'bank_amount': bank_row['amount'],
                'bank_date': bank_row['posting_date'],
                'confidence_score': round(confidence, 2),
                'narration_score': round(narr_score * 100, 2),
                'amount_score': round(amt_score * 100, 2),
                'date_score': round(date_score * 100, 2),
                'customer_score': cust_score * 100
            })
            
    except Exception as e:
        print(f"⚠ Warning: Error processing payment {pay_row.get('payment_ref', 'unknown')}: {e}")
        continue

if not results:
    print("✗ No matches found!")
    sys.exit(1)

matches = pd.DataFrame(results)
print(f"✓ Generated {len(matches)} potential matches")

# -----------------------------
# 5. One-to-One Matching Constraint
# -----------------------------
# Sort by confidence and apply greedy one-to-one matching
matches_sorted = matches.sort_values('confidence_score', ascending=False).reset_index(drop=True)

best_matches = []
matched_payments = set()
matched_bank_txns = set()

for _, match in matches_sorted.iterrows():
    pay_ref = match['payment_ref']
    bank_ref = match['bank_txn_ref']
    
    # Only add if neither side has been matched yet
    if pay_ref not in matched_payments and bank_ref not in matched_bank_txns:
        best_matches.append(match)
        matched_payments.add(pay_ref)
        matched_bank_txns.add(bank_ref)

best_matches = pd.DataFrame(best_matches)
print(f"✓ Selected {len(best_matches)} best one-to-one matches")

# -----------------------------
# 6. Classification for Audit
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
# 7. Create Unmatched Reports
# -----------------------------
# Unmatched payments
unmatched_payments = payments[~payments['payment_ref'].isin(best_matches['payment_ref'])].copy()
unmatched_payments['reason'] = 'No suitable bank transaction found'
print(f"⚠ {len(unmatched_payments)} unmatched payments")

# Unmatched bank transactions
unmatched_bank = bank[~bank['bank_txn_ref'].isin(best_matches['bank_txn_ref'])].copy()
unmatched_bank['reason'] = 'No matching payment record found'
print(f"⚠ {len(unmatched_bank)} unmatched bank transactions")

# -----------------------------
# 8. Audit-Friendly Outputs with Timestamps
# -----------------------------
try:
    # Main reconciliation results
    best_matches.to_excel(out_dir / 'reconciliation_results_detailed.xlsx', index=False)
    print(f"✓ Saved: reconciliation_results_detailed.xlsx")
    
    # Summary statistics
    summary = best_matches.groupby('recon_status').agg(
        transaction_count=('payment_ref', 'count'),
        total_amount=('payment_amount', 'sum'),
        avg_confidence=('confidence_score', 'mean')
    ).reset_index()
    summary.to_excel(out_dir / 'reconciliation_summary.xlsx', index=False)
    print(f"✓ Saved: reconciliation_summary.xlsx")
    
    # Unmatched payments report
    if not unmatched_payments.empty:
        unmatched_payments.to_excel(out_dir / 'unmatched_payments.xlsx', index=False)
        print(f"✓ Saved: unmatched_payments.xlsx ({len(unmatched_payments)} records)")
    
    # Unmatched bank transactions report
    if not unmatched_bank.empty:
        unmatched_bank.to_excel(out_dir / 'unmatched_bank_transactions.xlsx', index=False)
        print(f"✓ Saved: unmatched_bank_transactions.xlsx ({len(unmatched_bank)} records)")
    
    # Reconciliation metadata
    metadata = pd.DataFrame([{
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_payments': len(payments),
        'total_bank_txns': len(bank),
        'matched_count': len(best_matches),
        'unmatched_payments': len(unmatched_payments),
        'unmatched_bank_txns': len(unmatched_bank),
        'auto_reconciled': len(best_matches[best_matches['recon_status'] == 'Auto-Reconciled']),
        'needs_review': len(best_matches[best_matches['recon_status'] == 'Review Recommended']),
        'match_rate': f"{len(best_matches) / len(payments) * 100:.1f}%"
    }])
    metadata.to_excel(out_dir / 'reconciliation_metadata.xlsx', index=False)
    print(f"✓ Saved: reconciliation_metadata.xlsx")
    
    print(f"\n{'='*60}")
    print(f"RECONCILIATION COMPLETE")
    print(f"{'='*60}")
    print(f"Matched: {len(best_matches)} / {len(payments)} payments ({len(best_matches)/len(payments)*100:.1f}%)")
    print(f"Auto-reconciled: {len(best_matches[best_matches['recon_status'] == 'Auto-Reconciled'])}")
    print(f"Needs review: {len(best_matches[best_matches['recon_status'] == 'Review Recommended'])}")
    print(f"Unmatched payments: {len(unmatched_payments)}")
    print(f"Unmatched bank txns: {len(unmatched_bank)}")
    print(f"\nAll outputs saved to: {out_dir}")
    
except Exception as e:
    print(f"✗ Error saving outputs: {e}")
    sys.exit(1)

# Display results
best_matches.head()