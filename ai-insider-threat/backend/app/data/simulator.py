import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

USER_ROLES = ['Admin', 'Engineer', 'HR', 'Contractor']
USERS = [f'User_{i}' for i in range(1, 21)]
USER_METADATA = {u: random.choice(USER_ROLES) for u in USERS}

EMAIL_TEXTS_NORMAL = [
    "Meeting agenda for tomorrow",
    "Please review the attached document",
    "Lunch at 12?",
    "Deployment successful",
    "Weekly status report"
]
EMAIL_TEXTS_SUSPICIOUS = [
    "Resignation letter and company IP",
    "Confidential source code transfer",
    "Customer database export",
    "Password list attached",
    "Urgent wire transfer request"
]

def generate_normal_logs(num_logs=200):
    logs = []
    base_time = datetime.now() - timedelta(days=7)
    for _ in range(num_logs):
        user = random.choice(USERS)
        # Normal hours: 8 AM to 6 PM
        hour = random.randint(8, 17)
        log_time = base_time + timedelta(minutes=random.randint(0, 10000))
        log_time = log_time.replace(hour=hour)
        
        event_type = random.choice(['login', 'file_access', 'usb_connect', 'email'])
        log = {
            'timestamp': log_time.isoformat(),
            'user': user,
            'role': USER_METADATA[user],
            'event_type': event_type,
            'details': ''
        }
        if event_type == 'file_access':
            log['details'] = f"Accessed file: /docs/{random.choice(['public', 'team', 'guidelines'])}/doc_{random.randint(1,10)}.pdf"
            log['file_size_mb'] = random.uniform(0.1, 5.0)
        elif event_type == 'usb_connect':
            log['details'] = 'Connected Kingston DataTraveler 16GB'
        elif event_type == 'email':
            log['details'] = random.choice(EMAIL_TEXTS_NORMAL)
        
        logs.append(log)
    return logs

def inject_red_team_logs(num_logs=10):
    logs = []
    base_time = datetime.now() - timedelta(days=2)
    malicious_users = random.sample(USERS, 2)
    for _ in range(num_logs):
        user = random.choice(malicious_users)
        # Unusual hours: 12 AM to 4 AM
        hour = random.randint(0, 4)
        log_time = base_time + timedelta(minutes=random.randint(0, 1000))
        log_time = log_time.replace(hour=hour)
        
        event_type = random.choice(['file_access', 'usb_connect', 'email'])
        log = {
            'timestamp': log_time.isoformat(),
            'user': user,
            'role': USER_METADATA[user],
            'event_type': event_type,
            'details': '',
            'is_malicious_simulated': True
        }
        if event_type == 'file_access':
            log['details'] = f"Accessed file: /confidential/source_code/main_{random.randint(1,5)}.py"
            log['file_size_mb'] = random.uniform(500.0, 2000.0) # Large download
        elif event_type == 'usb_connect':
            log['details'] = 'Connected Unknown Mass Storage Device 1TB'
        elif event_type == 'email':
            log['details'] = random.choice(EMAIL_TEXTS_SUSPICIOUS)
        
        logs.append(log)
    return logs

def get_simulated_data():
    normal = generate_normal_logs(300)
    red_team = inject_red_team_logs(15)
    all_logs = normal + red_team
    # sort by timestamp
    all_logs.sort(key=lambda x: x['timestamp'])
    return pd.DataFrame(all_logs)
