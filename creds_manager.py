import json
import os
from typing import Dict, Any

CREDS_DIR = "user_creds"  # Or use a database later

def save_user_creds(user_uuid: str, servers: dict):
    """Save user credentials securely"""
    os.makedirs(CREDS_DIR, exist_ok=True)
    filepath = f"{CREDS_DIR}/{user_uuid}.json"
    
    # In production, encrypt this!
    with open(filepath, 'w') as f:
        json.dump(servers, f)

def get_user_servers(user_uuid: str) -> dict:
    """Get user's server credentials"""
    filepath = f"{CREDS_DIR}/{user_uuid}.json"
    
    if not os.path.exists(filepath):
        raise ValueError(f"No credentials found for user {user_uuid}")
    
    with open(filepath, 'r') as f:
        return json.load(f)