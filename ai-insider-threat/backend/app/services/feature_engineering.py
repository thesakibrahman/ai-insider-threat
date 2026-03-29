import pandas as pd
import numpy as np
from app.services.llm_service import analyze_text_intent

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms preprocessed logs into a feature matrix suitable for ML models.
    Creates an event-level feature row per log entry.

    Features extracted (as per paper's Preprocessing Layer):
      - Temporal:        off-hours flag, weekend flag, hour of day
      - Event type:      login, file_access, usb_connect, email (one-hot style)
      - File:            log-scaled file size, USB large-transfer flag
      - Frequency:       per-user login count (login frequency, cited in methodology)
      - NLP / LLM:       intent score for email, file_access, and login events
                         (paper: 'emails, file activity, and HTTP requests')
    """
    if df.empty:
        return df

    # Pre-compute per-user login frequency for the entire batch
    # (Methodology explicitly mentions "login frequency" as a key feature)
    login_counts = (
        df[df['event_type'] == 'login']
        .groupby('user')
        .size()
        .to_dict()
    )

    features = []

    for idx, row in df.iterrows():
        record = {
            'log_id':     idx,
            'user':       row['user'],
            'timestamp':  row['timestamp'],
            'is_malicious': row.get('is_malicious_simulated', False),
            'event_type': row['event_type']
        }

        # ── 1. Temporal / Frequency Features ────────────────────────────────
        # Off-hours: outside 8 AM – 6 PM (strong insider-threat signal)
        hour = row.get('hour', 0)
        record['feat_off_hours']   = 1 if (hour < 8 or hour > 18) else 0
        record['feat_is_weekend']  = row.get('is_weekend', 0)
        record['feat_hour_of_day'] = int(hour)  # raw hour for model

        # ── 2. Behavioral / Event-Type Features ─────────────────────────────
        event = row.get('event_type', '')
        record['feat_is_login']       = 1 if event == 'login'       else 0
        record['feat_is_file_access'] = 1 if event == 'file_access' else 0
        record['feat_is_usb']         = 1 if event == 'usb_connect' else 0
        record['feat_is_email']       = 1 if event == 'email'        else 0

        # ── 3. File / Device Anomaly Features ───────────────────────────────
        file_size = float(row.get('file_size_mb', 0) or 0)
        record['feat_file_size'] = np.log1p(file_size)   # log-scaled
        # USB large-transfer flag (>100 MB is suspicious)
        record['feat_usb_large'] = 1 if (event == 'usb_connect' and file_size > 100) else 0

        # ── 4. Login Frequency Feature ───────────────────────────────────────
        # Number of logins by this user in the current batch
        # (Methodology: 'login frequency' is an extracted feature)
        record['feat_login_count'] = login_counts.get(row.get('user', ''), 0)

        # ── 5. NLP / LLM Intent Score ────────────────────────────────────────
        # Paper: "LLMs fine-tuned on emails, file activity, and HTTP requests"
        # We apply the intent scorer to all text-bearing event types.
        intent_score = 0.0
        details = str(row.get('details', ''))
        if event in ('email', 'file_access', 'login') and details:
            intent_score = analyze_text_intent(details)
        record['feat_nlp_intent_score'] = intent_score

        features.append(record)

    features_df = pd.DataFrame(features)
    return features_df


def get_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts only the numerical feature columns (prefixed 'feat_') for model input.
    """
    if features_df.empty:
        return features_df
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    return features_df[feature_cols].copy()