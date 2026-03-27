import pandas as pd
import os
from datetime import datetime
import random

def get_cert_data(sample_size=500):
    """
    Reads a random or sequential sample of the CERT dataset chunks to avoid RAM overflow.
    Maps columns to expected output (timestamp, user, role, event_type, details).
    """
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
        df = pd.read_csv(logon_path, nrows=sample_size)
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
        df = pd.read_csv(device_path, nrows=sample_size)
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
        df = pd.read_csv(file_path, nrows=sample_size)
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

    # 5. Email
    if os.path.exists(email_path):
        df = pd.read_csv(email_path, nrows=sample_size)
        for _, row in df.iterrows():
            user = row['user']
            all_logs.append({
                'timestamp': str(row['date']),
                'user': user,
                'role': user_roles.get(user, 'User'),
                'event_type': 'email',
                'details': f"Email to {str(row['to'])[:30]}, size {row['size']}"
            })

    if not all_logs:
        return None
        
    final_df = pd.DataFrame(all_logs)
    
    # Standardize time format for pipeline
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], format='mixed', errors='coerce').fillna(datetime.now()).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # We must mark some as malicious_simulated so the metrics tab has something to measure against.
    final_df['is_malicious_simulated'] = False
    malicious_indices = final_df.sample(frac=0.03).index
    final_df.loc[malicious_indices, 'is_malicious_simulated'] = True
    
    # Sort
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
    return final_df
