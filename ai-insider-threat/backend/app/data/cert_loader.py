import pandas as pd
import os
from datetime import datetime
import random
from sklearn.model_selection import train_test_split

def get_cert_data(sample_size=500, split=False):
    """
    Reads a sample of the CERT dataset and maps columns to the expected pipeline format.
    
    Args:
        sample_size: Number of rows to read from each source CSV file.
        split: If True, returns (train_df, val_df, test_df) split 70/15/15.
               If False, returns the full DataFrame (legacy behaviour).
    """
    # We intentionally sample MORE rows than sample_size so that after the 70/15/15 split
    # the training set still contains at least `sample_size` events.
    _load_size = int(sample_size / 0.70) + 1  # ensures train split ~ sample_size
    base_dir = os.path.join(os.path.dirname(__file__), 'CERT Data')
    
    if not os.path.exists(base_dir):
        return None
        
    users_path = os.path.join(base_dir, 'users.csv')
    logon_path = os.path.join(base_dir, 'logon.csv')
    device_path = os.path.join(base_dir, 'device.csv')
    file_path = os.path.join(base_dir, 'file.csv')
    email_path = os.path.join(base_dir, 'email.csv')
    
    # 1. Load users to map roles
    user_roles = {}
    if os.path.exists(users_path):
        users_df = pd.read_csv(users_path)
        for _, row in users_df.iterrows():
            user_roles[row['user_id']] = row['role']
            
    all_logs = []
    
    # 2. Logon
    if os.path.exists(logon_path):
        df = pd.read_csv(logon_path, nrows=_load_size)
        for _, row in df.iterrows():
            user = row['user']
            all_logs.append({
                'timestamp': str(row['date']),
                'user': user,
                'role': user_roles.get(user, 'User'),
                'event_type': 'login',
                'details': f"Activity: {row['activity']} on PC: {row['pc']}"
            })

    # 3. Device
    if os.path.exists(device_path):
        df = pd.read_csv(device_path, nrows=_load_size)
        for _, row in df.iterrows():
            user = row['user']
            all_logs.append({
                'timestamp': str(row['date']),
                'user': user,
                'role': user_roles.get(user, 'User'),
                'event_type': 'usb_connect',
                'details': f"Device {row['activity']}. Tree: {str(row.get('file_tree', ''))[:50]}"
            })

    # 4. File
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, nrows=_load_size)
        for _, row in df.iterrows():
            user = row['user']
            all_logs.append({
                'timestamp': str(row['date']),
                'user': user,
                'role': user_roles.get(user, 'User'),
                'event_type': 'file_access',
                'file_size_mb': random.uniform(0.1, 5.0), # mock size metric for AE
                'details': f"{row['activity']} -> {str(row['filename'])[:50]}"
            })

    # 5. Email — include subject/content so LLM intent scorer has real text
    if os.path.exists(email_path):
        df = pd.read_csv(email_path, nrows=_load_size)
        for _, row in df.iterrows():
            user = row['user']
            # Try to get meaningful text for LLM analysis
            # CERT v6.2 has: to, cc, bcc, from, size, attachments, content
            subject_text = str(row.get('content', '') or '')[:120]
            if not subject_text.strip():
                subject_text = f"Email to {str(row.get('to', 'unknown'))[:30]}, size {row.get('size', 0)} bytes"
            all_logs.append({
                'timestamp': str(row['date']),
                'user': user,
                'role': user_roles.get(user, 'User'),
                'event_type': 'email',
                'details': subject_text
            })

    if not all_logs:
        return None
        
    final_df = pd.DataFrame(all_logs)
    
    # Standardize time format for pipeline
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], format='mixed', errors='coerce').fillna(datetime.now()).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Mark 3% as malicious (simulated ground-truth labels for metrics)
    final_df['is_malicious_simulated'] = False
    malicious_indices = final_df.sample(frac=0.03, random_state=42).index
    final_df.loc[malicious_indices, 'is_malicious_simulated'] = True
    
    # Sort by time (chronological order — important for realistic streaming splits)
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Train / Validation / Test split: 70% / 15% / 15%                   #
    # As stated in the paper's Experimental Setup section.                #
    #                                                                      #
    # We split CHRONOLOGICALLY (not randomly) because in real deployments  #
    # the model always sees past data and is tested on future events.      #
    # ------------------------------------------------------------------ #
    if not split:
        # Legacy: return the full dataset (used during prototype/demo mode)
        return final_df

    total = len(final_df)
    train_end = int(total * 0.70)   # first 70% — training
    val_end   = int(total * 0.85)   # next  15% — validation
    # remaining 15% — test

    train_df = final_df.iloc[:train_end].reset_index(drop=True)
    val_df   = final_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = final_df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df
