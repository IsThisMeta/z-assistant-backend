import json
import hashlib
import hmac
import base64
import httpx
from typing import Dict, Any, Optional

# In-memory credential store (cleared after each request)
# Format: {device_id: {hmac_key: str, decrypted_servers: dict}}
_credential_cache: Dict[str, Dict[str, Any]] = {}

def decrypt_credentials(encrypted_data: dict, hmac_key: str) -> dict:
    """Decrypt HMAC-protected credentials"""
    decrypted = {}

    for service, encrypted in encrypted_data.items():
        try:
            # Split the encrypted data and HMAC
            parts = encrypted.split('::')
            if len(parts) != 2:
                continue

            xor_encrypted, signature = parts

            # Verify HMAC
            expected_hmac = hmac.new(
                hmac_key.encode(),
                xor_encrypted.encode(),
                hashlib.sha256
            ).hexdigest()

            if signature != expected_hmac:
                continue

            # Decrypt XOR
            encrypted_bytes = base64.b64decode(xor_encrypted)
            key_bytes = hmac_key.encode()
            decrypted_bytes = bytes([
                encrypted_bytes[i] ^ key_bytes[i % len(key_bytes)]
                for i in range(len(encrypted_bytes))
            ])

            # Parse JSON
            decrypted[service] = json.loads(decrypted_bytes.decode())

        except Exception as e:
            print(f"Failed to decrypt {service}: {e}")
            continue

    return decrypted

def store_session_creds(device_id: str, hmac_key: str, encrypted_servers: dict):
    """Store encrypted credentials and HMAC key for this session"""
    _credential_cache[device_id] = {
        'hmac_key': hmac_key,
        'encrypted_servers': encrypted_servers
    }

def clear_session_creds(device_id: str):
    """Clear credentials from memory after request"""
    if device_id in _credential_cache:
        del _credential_cache[device_id]

def _get_decrypted_servers(device_id: str) -> dict:
    """Internal: Decrypt and return server credentials"""
    if device_id not in _credential_cache:
        raise ValueError("No credentials in session")

    cache = _credential_cache[device_id]
    hmac_key = cache['hmac_key']
    encrypted = cache['encrypted_servers']

    # Decrypt on-the-fly
    return decrypt_credentials(encrypted, hmac_key)

# NO SECURE WRAPPER FUNCTIONS
# Backend never calls user servers - all operations happen on device
# This ensures zero IP logging and complete privacy