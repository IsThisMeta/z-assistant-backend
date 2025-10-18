from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, validator
from openai import OpenAI
import httpx
import os
from typing import Dict, Any, List, Optional
import uuid
import json
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
from collections import defaultdict
import re
import time
import logging
from upstash_redis import Redis
import hashlib
import hmac
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

@app.middleware("http")
async def mask_request_logging(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    client_identifier = request.headers.get("X-Device-Id") or request.headers.get("X-User-Id")
    if client_identifier:
        if len(client_identifier) > 8:
            client_identifier = f"{client_identifier[:8]}..."
    else:
        client_identifier = "masked"

    logger.info(
        "HTTP %s %s -> %d (%.2f ms) client=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        client_identifier,
    )

    return response

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase client (for auth - will use user's token for RLS)
supabase: Client = create_client(
    os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Only for admin operations
)

# Get API keys from environment
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY environment variable is required")
REVENUECAT_SECRET_KEY = os.getenv("REVENUECAT_SECRET_KEY")

# Initialize Upstash Redis for distributed rate limiting
redis_client = None
try:
    upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

    if upstash_url and upstash_token:
        redis_client = Redis(url=upstash_url, token=upstash_token)
        logger.info("✅ Upstash Redis initialized for rate limiting")
    else:
        logger.warning("⚠️  Upstash credentials not found - rate limiting disabled")
except Exception as e:
    logger.error(f"⚠️  Failed to initialize Upstash Redis: {e}")
    redis_client = None

# Rate limiting configuration - Tier-based
RATE_LIMIT_PRO_REQUESTS = 3  # Pro: 3 messages per 12 hours
RATE_LIMIT_MEGA_REQUESTS = 15  # Mega: 15 messages per 12 hours
RATE_LIMIT_WINDOW = 43200  # 12 hours in seconds
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "300"))  # default 5 minutes

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None  # Optional context like "explore"
    # NO SERVERS - Zero-knowledge architecture!
    # Backend uses library_cache from Supabase instead

# Security Functions
async def verify_mega_subscription(authorization: str = Header(None)) -> str:
    """Verify user has active Mega subscription and return user_id"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Extract bearer token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use: Bearer <token>")

    token = authorization.replace("Bearer ", "")

    try:
        # TEST MODE: Allow "test-bypass" token for development
        if token == "test-bypass":
            logger.warning("⚠️  TEST MODE: Bypassing authentication with test user")
            return "test-user-rate-limit"

        # Verify token with Supabase
        user_response = supabase.auth.get_user(token)
        user = user_response.user if hasattr(user_response, 'user') else user_response

        if not user or not user.id:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user_id = user.id

        # Check if user has active Mega subscription using RPC
        # Clean 1.2: Function always returns boolean, no NULL handling needed
        result = supabase.rpc('has_active_mega', params={'p_user_id': user_id}).execute()
        has_mega = bool(result.data)

        if not has_mega:
            raise HTTPException(
                status_code=403,
                detail="Zagreus Mega subscription required. Upgrade in Settings > Subscriptions.",
            )

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        print(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Device-based authentication with HMAC
async def verify_device_subscription(x_device_id: str = Header(None)) -> tuple[str, str]:
    """Verify device has active Mega subscription and return (device_id, hmac_key)"""
    if not x_device_id:
        raise HTTPException(status_code=401, detail="X-Device-Id header required")

    try:
        # Validate UUID format
        device_uuid = uuid.UUID(x_device_id)
        device_id = str(device_uuid)

        # Look up device in database
        result = supabase.table('device_keys').select('hmac_key').eq('device_id', device_id).execute()

        if not result.data:
            raise HTTPException(status_code=403, detail="Device not registered. Please update the app.")

        hmac_key = result.data[0]['hmac_key']

        # Update last_used timestamp
        supabase.table('device_keys').update({
            'last_used': datetime.now().isoformat()
        }).eq('device_id', device_id).execute()

        return device_id, hmac_key

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid device ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Device auth error: {e}")
        raise HTTPException(status_code=401, detail="Device authentication failed")

def decrypt_credentials(encrypted_data: dict, hmac_key: str) -> dict:
    """Decrypt HMAC-protected credentials"""
    decrypted = {}

    for service, encrypted in encrypted_data.items():
        try:
            # Split the encrypted data and HMAC
            parts = encrypted.split('::')
            if len(parts) != 2:
                logger.warning(f"Invalid encrypted format for {service}")
                continue

            xor_encrypted, signature = parts

            # Verify HMAC
            expected_hmac = hmac.new(
                hmac_key.encode(),
                xor_encrypted.encode(),
                hashlib.sha256
            ).hexdigest()

            if signature != expected_hmac:
                logger.warning(f"HMAC verification failed for {service}")
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
            logger.error(f"Failed to decrypt {service}: {e}")
            continue

    return decrypted

async def check_rate_limit(device_id: str):
    """Check and enforce tier-based rate limits per device using Upstash Redis"""
    if not redis_client:
        # Redis not available - skip rate limiting (dev mode)
        print(f"⚠️  Rate limiting skipped for {device_id[:8]}... (Redis not configured)")
        return

    # Get subscription tier for this device
    try:
        result = supabase.rpc('get_device_subscription_tier', params={'p_device_id': device_id}).execute()
        tier = result.data if result.data else None

        # Determine rate limit based on tier
        if tier == 'mega':
            limit = RATE_LIMIT_MEGA_REQUESTS
            tier_name = "Mega (15 messages/12 hours)"
        elif tier == 'pro':
            limit = RATE_LIMIT_PRO_REQUESTS
            tier_name = "Pro (3 messages/12 hours)"
        else:
            # No active subscription - shouldn't happen as auth checks this
            raise HTTPException(
                status_code=403,
                detail="No active Pro or Mega subscription found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subscription tier for device {device_id[:8]}...: {e}")
        # Default to Pro tier on error (more restrictive)
        limit = RATE_LIMIT_PRO_REQUESTS
        tier_name = "Pro (fallback)"

    # Use Redis sorted set for time-based rate limiting
    # Key format: ratelimit:{device_id}
    # Score: timestamp
    # Value: unique request ID
    key = f"ratelimit:{device_id}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    try:
        # Remove old entries outside the time window
        redis_client.zremrangebyscore(key, 0, window_start)

        # Count requests in current window
        count = redis_client.zcard(key)

        # Check if limit exceeded
        if count >= limit:
            # Get the oldest request timestamp
            oldest = redis_client.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = int(oldest_time + RATE_LIMIT_WINDOW - now)
                hours = retry_after // 3600
                minutes = (retry_after % 3600) // 60

                time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {tier_name}. Try again in {time_str}.",
                    headers={"Retry-After": str(retry_after)}
                )

        # Add current request with timestamp as score
        request_id = f"{now}:{uuid.uuid4()}"
        redis_client.zadd(key, {request_id: now})

        # Set expiry on the key (cleanup after window expires)
        redis_client.expire(key, RATE_LIMIT_WINDOW)

        print(f"✅ Rate limit check passed for {device_id[:8]}... [{tier_name}]: {count + 1}/{limit}")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Redis error - log but don't block request
        print(f"⚠️  Rate limit check failed for {device_id[:8]}...: {e}")
        # Continue without rate limiting on Redis errors

def validate_url(url: str) -> bool:
    """Validate URL format"""
    return bool(re.match(r'^https?://[^\s]+$', url))

# Zero-knowledge architecture: Backend never calls user servers
# All add/delete/update operations happen on device

# Function for Responses API
def search_movies_in_library(query: str, device_id: str) -> Dict[str, Any]:
    """Search for movies already in your library from Supabase cache"""
    print(f"🔧 TOOL CALLED: search_movies_in_library - Query: {query} (device: {device_id[:8]}...)")

    try:
        # Read from Supabase library_cache - ZERO-KNOWLEDGE!
        # Note: Removed retry logic to avoid blocking - if sync in progress, return immediately
        result = supabase.table('library_cache').select('movies, synced_at, is_syncing').eq('device_id', device_id).execute()

        if not result.data:
            return {"error": "Library not synced yet. Please sync your library first."}

        cache = result.data[0]

        # Check if sync is in progress
        if cache.get('is_syncing'):
            return {"error": "Library sync in progress. Please try again in a moment."}

            # Sync complete - search for matches
            all_movies = cache.get('movies', [])
            query_lower = query.lower()

            # Search for matches
            matches = []
            for movie in all_movies:
                if query_lower in movie.get("title", "").lower():
                    matches.append({
                        "title": movie["title"],
                        "year": movie.get("year"),
                        "has_file": movie.get("has_file", False),
                        "tmdb_id": movie.get("tmdb_id"),
                        "quality": movie.get("quality")
                    })

            if not matches:
                return {"matches": [], "message": f"No movies found matching '{query}'"}

            print(f"  ✓ Found {len(matches)} matches")
            return {"matches": matches[:10]}  # Limit to 10 results

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"error": str(e)}

# Removed search_media - dead code that violated zero-knowledge architecture

# Function for Responses API
def get_radarr_quality_profiles(device_id: str) -> Dict[str, Any]:
    """Return cached Radarr quality profiles from the device library sync."""
    print(f"🔧 TOOL CALLED: get_radarr_quality_profiles (device: {device_id[:8]}...)")

    try:
        result = supabase.table('library_cache').select('radarr_profiles, synced_at, is_syncing').eq('device_id', device_id).execute()

        if not result.data:
            return {"error": "Library not synced yet. Please sync Radarr first."}

        cache = result.data[0]

        if cache.get('is_syncing'):
            return {"error": "Library sync in progress. Try again in a few seconds."}

        profiles = cache.get('radarr_profiles') or []
        print(f"  ✓ Retrieved {len(profiles)} Radarr profiles")
        return {"profiles": profiles, "total": len(profiles)}

    except Exception as exc:
        print(f"  ✗ Error: {exc}")
        return {"error": str(exc)}


def get_sonarr_quality_profiles(device_id: str) -> Dict[str, Any]:
    """Return cached Sonarr quality profiles from the device library sync."""
    print(f"🔧 TOOL CALLED: get_sonarr_quality_profiles (device: {device_id[:8]}...)")

    try:
        result = supabase.table('library_cache').select('sonarr_profiles, synced_at, is_syncing').eq('device_id', device_id).execute()

        if not result.data:
            return {"error": "Library not synced yet. Please sync Sonarr first."}

        cache = result.data[0]

        if cache.get('is_syncing'):
            return {"error": "Library sync in progress. Try again in a few seconds."}

        profiles = cache.get('sonarr_profiles') or []
        print(f"  ✓ Retrieved {len(profiles)} Sonarr profiles")
        return {"profiles": profiles, "total": len(profiles)}

    except Exception as exc:
        print(f"  ✗ Error: {exc}")
        return {"error": str(exc)}


# Function for Responses API
def get_show_episodes(show_title: str, device_id: str) -> Dict[str, Any]:
    """Request episode details for a show from device - ZERO-KNOWLEDGE on-demand fetch!"""
    print(f"🔧 TOOL CALLED: get_show_episodes - Show: {show_title} (device: {device_id[:8]}...)")

    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Stage fetch command for device to pick up
        print(f"  → Staging episode fetch request: {request_id}")
        supabase.table('data_fetch_commands').insert({
            'request_id': request_id,
            'device_id': device_id,
            'action': 'fetch_episodes',
            'show_title': show_title,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }).execute()

        # Wait for device to complete fetch (3x5s retry pattern)
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            time.sleep(retry_delay)

            # Check command status
            cmd_result = supabase.table('data_fetch_commands').select('status').eq('request_id', request_id).execute()

            if cmd_result.data and cmd_result.data[0]['status'] == 'completed':
                # Device completed - read episode cache
                print(f"  ✓ Device completed fetch, reading episode cache...")
                cache_result = supabase.table('episode_cache').select('episodes').eq('device_id', device_id).eq('show_title', show_title).execute()

                if cache_result.data:
                    episodes = cache_result.data[0]['episodes']
                    print(f"  ✓ Retrieved {len(episodes)} episodes for {show_title}")
                    return {"episodes": episodes, "show_title": show_title}
                else:
                    return {"error": f"No episode data found for {show_title}"}

            if attempt < max_retries - 1:
                print(f"  → Waiting for device response... retry {attempt + 2}/{max_retries}")

        # Timeout
        print(f"  ✗ Device did not respond in time")
        return {"error": "Request timed out - device may be offline or busy"}

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"error": str(e)}

# Function for Responses API
def get_all_shows(device_id: str) -> Dict[str, Any]:
    """Get list of all TV shows in the library from Supabase cache"""
    print(f"🔧 TOOL CALLED: get_all_shows (device: {device_id[:8]}...)")
    try:
        # Retry up to 3 times if sync is in progress
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            # Read from Supabase library_cache
            result = supabase.table('library_cache').select('shows, synced_at, is_syncing').eq('device_id', device_id).execute()

            if not result.data:
                print("  → No cache found - library may be syncing for first time")
                if attempt < max_retries - 1:
                    print(f"  → Waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                return {"error": "Library not synced yet. Please try again later."}

            cache = result.data[0]

            # Check if sync is in progress
            if cache.get('is_syncing'):
                if attempt < max_retries - 1:
                    print(f"  → Sync in progress, waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("  ✗ Sync still in progress after 3 retries")
                    return {"error": "Library sync is taking longer than expected. Please try again in a moment."}

            # Sync complete - return shows
            synced_at = datetime.fromisoformat(cache['synced_at'])
            age_hours = (datetime.now(synced_at.tzinfo) - synced_at).total_seconds() / 3600

            # Check if cache is stale (> 24 hours)
            if age_hours > 24:
                print(f"  → Cache is {age_hours:.1f}h old - requesting refresh")
                return {
                    "error": "Library cache is stale. Syncing...",
                    "requires_sync": True
                }

            shows = cache.get('shows', [])
            sample_count = min(len(shows), 10)
            if sample_count:
                print(f"  ✓ Retrieved {len(shows)} shows from cache (showing first {sample_count} in logs, synced {age_hours:.1f}h ago)")
            else:
                print("  ✓ Retrieved 0 shows from cache")

            return {"shows": shows, "total": len(shows)}

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"error": str(e)}

# Function for Responses API
def get_library_stats(device_id: str) -> Dict[str, Any]:
    """Get statistics about the user's media library from Supabase cache"""
    print(f"🔧 TOOL CALLED: get_library_stats (device: {device_id[:8]}...)")

    try:
        # Retry up to 3 times if sync is in progress
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            # Read from Supabase library_cache - ZERO-KNOWLEDGE!
            result = supabase.table('library_cache').select('movies, shows, synced_at, is_syncing').eq('device_id', device_id).execute()

            if not result.data:
                print("  → No cache found - library may be syncing for first time")
                if attempt < max_retries - 1:
                    print(f"  → Waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                return {"error": "Library not synced yet. Please try again later."}

            cache = result.data[0]

            # Check if sync is in progress
            if cache.get('is_syncing'):
                if attempt < max_retries - 1:
                    print(f"  → Sync in progress, waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("  ✗ Sync still in progress after 3 retries")
                    return {"error": "Library sync is taking longer than expected. Please try again in a moment."}

            # Sync complete - return stats
            synced_at = datetime.fromisoformat(cache['synced_at'])
            age_hours = (datetime.now(synced_at.tzinfo) - synced_at).total_seconds() / 3600

            # Check if cache is stale (> 24 hours)
            if age_hours > 24:
                print(f"  ⚠️  Cache is {age_hours:.1f} hours old - requesting refresh")
                stats = {
                    "movies": len(cache.get('movies', [])),
                    "shows": len(cache.get('shows', [])),
                    "cache_age_hours": round(age_hours, 1),
                    "requires_sync": True
                }
                return stats

            stats = {
                "movies": len(cache.get('movies', [])),
                "shows": len(cache.get('shows', [])),
                "cache_age_hours": round(age_hours, 1)
            }
            print(f"  ✓ {stats['movies']} movies, {stats['shows']} shows (cache age: {age_hours:.1f}h)")
            return stats

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"error": str(e)}

# Removed add_movie_to_radarr - device handles all execution now

# Function for Responses API
def get_all_movies(device_id: str) -> Dict[str, Any]:
    """Get list of all movies in the library from Supabase cache"""
    print(f"🔧 TOOL CALLED: get_all_movies (device: {device_id[:8]}...)")
    try:
        # Retry up to 3 times if sync is in progress
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            # Read from Supabase library_cache
            result = supabase.table('library_cache').select('movies, synced_at, is_syncing').eq('device_id', device_id).execute()

            if not result.data:
                print("  → No cache found - library may be syncing for first time")
                if attempt < max_retries - 1:
                    print(f"  → Waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                return {"error": "Library not synced yet. Please try again later."}

            cache = result.data[0]

            # Check if sync is in progress
            if cache.get('is_syncing'):
                if attempt < max_retries - 1:
                    print(f"  → Sync in progress, waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print("  ✗ Sync still in progress after 3 retries")
                    return {"error": "Library sync is taking longer than expected. Please try again in a moment."}

            # Sync complete - return movies
            synced_at = datetime.fromisoformat(cache['synced_at'])
            age_hours = (datetime.now(synced_at.tzinfo) - synced_at).total_seconds() / 3600

            # Check if cache is stale (> 24 hours)
            if age_hours > 24:
                print(f"  → Cache is {age_hours:.1f}h old - requesting refresh")
                return {
                    "error": "Library cache is stale. Syncing...",
                    "requires_sync": True
                }

            movies = cache.get('movies', [])
            sample_count = min(len(movies), 20)
            if sample_count:
                print(f"  ✓ Retrieved {len(movies)} movies from cache (showing first {sample_count} in logs, synced {age_hours:.1f}h ago)")
            else:
                print("  ✓ Retrieved 0 movies from cache")

            return {"movies": movies, "total": len(movies)}

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"error": str(e)}

# Function for Responses API
def search_movies(query: str) -> Dict[str, Any]:
    """Search for movies on TMDB with light normalization and year hinting."""
    print(f"🔧 TOOL CALLED: search_movies - Query: {query}")

    def _normalize_title_and_year(raw: str) -> tuple[str, Optional[int]]:
        # Extract a 4-digit year if present
        year_match = re.search(r"\b(19|20)\d{2}\b", raw)
        year_val: Optional[int] = int(year_match.group(0)) if year_match else None

        # If year present, keep text before that year as title; else use full
        title_part = raw[: year_match.start()].strip() if year_match else raw

        # Remove quotes and common noise words
        title_part = re.sub(r"[\"'`]+", "", title_part)
        title_part = re.sub(r"\b(film|movie|by)\b", " ", title_part, flags=re.IGNORECASE)

        # Collapse whitespace and trim
        title_part = re.sub(r"\s+", " ", title_part).strip()

        # As a guard, if normalization produced empty title, fall back to raw
        if not title_part:
            title_part = raw

        return title_part, year_val

    # Lightweight in-process cache to avoid duplicate TMDB calls during a chat
    # Keyed by (normalized_title, year)
    if not hasattr(search_movies, "_cache"):
        search_movies._cache = {}

    try:
        base_url = "https://api.themoviedb.org/3/search/movie"

        title, year_hint = _normalize_title_and_year(query)
        cache_key = (title.lower(), year_hint)

        if cache_key in search_movies._cache:
            cached = search_movies._cache[cache_key]
            print("  → using cached TMDB search")
            return cached

        def _tmdb_search(q: str, year: Optional[int]) -> Dict[str, Any]:
            params = {"api_key": TMDB_API_KEY, "query": q, "include_adult": False}
            if year is not None:
                params["year"] = year

            resp = httpx.get(base_url, params=params, timeout=5.0)
            if resp.status_code != 200:
                return {"movies": [], "error": f"API error: {resp.status_code}"}

            results = resp.json().get("results", [])

            movies = []
            for item in results[:5]:
                y = None
                if item.get("release_date"):
                    try:
                        y = int(item["release_date"][:4])
                    except:
                        pass

                movies.append({
                    "tmdb_id": item["id"],
                    "title": item.get("title", ""),
                    "year": y,
                    "poster_path": item.get("poster_path"),
                    "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else ""
                })

            return {"movies": movies, "total_found": len(results)}

        # First attempt: normalized title + year (if found)
        result = _tmdb_search(title, year_hint)

        # Fallback: if no hits and we had a year, try without year
        if result.get("total_found", 0) == 0 and year_hint is not None:
            result = _tmdb_search(title, None)

        # Cache and return
        search_movies._cache[cache_key] = result
        print(f"  ✓ Found {len(result.get('movies', []))} movie results")
        return result

    except Exception as e:
        return {"movies": [], "error": str(e)}

# Function for Responses API
def search_shows(query: str) -> Dict[str, Any]:
    """Search for TV shows on TMDB and return a list of results"""
    print(f"🔧 TOOL CALLED: search_shows - Query: {query}")
    
    try:
        search_url = "https://api.themoviedb.org/3/search/tv"
        params = {
            "api_key": TMDB_API_KEY,
            "query": query
        }
        
        response = httpx.get(search_url, params=params, timeout=5.0)
        if response.status_code == 200:
            results = response.json().get("results", [])
            
            # Return top 5 results with relevant info
            shows = []
            for item in results[:5]:
                year = None
                if item.get("first_air_date"):
                    try:
                        year = int(item["first_air_date"][:4])
                    except:
                        pass
                
                shows.append({
                    "tmdb_id": item["id"],
                    "title": item["name"],
                    "year": year,
                    "poster_path": item.get("poster_path"),
                    "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else ""
                })
            
            print(f"  ✓ Found {len(shows)} TV show results")
            return {"shows": shows, "total_found": len(results)}
        
        return {"shows": [], "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"shows": [], "error": str(e)}

# Function for Responses API
def search_person(query: str) -> Dict[str, Any]:
    """Search for people (actors, directors, etc.) on TMDB"""
    print(f"🔧 TOOL CALLED: search_person - Query: {query}")

    try:
        search_url = "https://api.themoviedb.org/3/search/person"
        params = {
            "api_key": TMDB_API_KEY,
            "query": query,
            "include_adult": False
        }

        response = httpx.get(search_url, params=params, timeout=5.0)
        if response.status_code == 200:
            results = response.json().get("results", [])

            # Return top 5 results with relevant info
            people = []
            for item in results[:5]:
                people.append({
                    "person_id": item["id"],
                    "name": item["name"],
                    "known_for_department": item.get("known_for_department", ""),
                    "profile_path": item.get("profile_path"),
                    "popularity": item.get("popularity", 0)
                })

            print(f"  ✓ Found {len(people)} people")
            return {"people": people, "total_found": len(results)}

        return {"people": [], "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"people": [], "error": str(e)}

# Function for Responses API
def get_person_credits(person_id: int) -> Dict[str, Any]:
    """Get all movie and TV credits for a person"""
    print(f"🔧 TOOL CALLED: get_person_credits - Person ID: {person_id}")

    try:
        credits_url = f"https://api.themoviedb.org/3/person/{person_id}/combined_credits"
        params = {
            "api_key": TMDB_API_KEY
        }

        response = httpx.get(credits_url, params=params, timeout=5.0)
        if response.status_code == 200:
            data = response.json()

            # Process movie credits
            movies = []
            for item in data.get("cast", []):
                if item.get("media_type") == "movie" and item.get("release_date"):
                    try:
                        year = int(item["release_date"][:4])
                        movies.append({
                            "tmdb_id": item["id"],
                            "title": item.get("title", ""),
                            "year": year,
                            "character": item.get("character", ""),
                            "poster_path": item.get("poster_path"),
                            "vote_average": item.get("vote_average", 0),
                            "popularity": item.get("popularity", 0)
                        })
                    except:
                        pass

            # Process TV credits
            shows = []
            for item in data.get("cast", []):
                if item.get("media_type") == "tv" and item.get("first_air_date"):
                    try:
                        year = int(item["first_air_date"][:4])
                        shows.append({
                            "tmdb_id": item["id"],
                            "title": item.get("name", ""),
                            "year": year,
                            "character": item.get("character", ""),
                            "poster_path": item.get("poster_path"),
                            "vote_average": item.get("vote_average", 0),
                            "popularity": item.get("popularity", 0)
                        })
                    except:
                        pass

            # Sort by year descending (most recent first)
            movies.sort(key=lambda x: x["year"], reverse=True)
            shows.sort(key=lambda x: x["year"], reverse=True)

            print(f"  ✓ Found {len(movies)} movies and {len(shows)} TV shows")
            return {
                "movies": movies,
                "shows": shows,
                "total_movies": len(movies),
                "total_shows": len(shows)
            }

        return {"movies": [], "shows": [], "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"movies": [], "shows": [], "error": str(e)}

# Function for Responses API
def add_to_queue(items: List[Dict], device_id: str = None) -> Dict[str, Any]:
    """Queue 1-3 items for instant execution on device (no modal)"""
    if not items:
        return {"error": "add_to_queue requires at least one item"}
    if len(items) > 3:
        return {"error": "add_to_queue supports at most 3 items. Use add_to_stage for larger batches."}
    return add_to_stage(operation="queue", items=items, device_id=device_id)

# Function for Responses API
def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_library_show_map(device_id: Optional[str]) -> Dict[int, int]:
    """Return a mapping of TMDB show IDs to TVDB IDs from the cached library."""
    if not device_id:
        return {}

    try:
        result = supabase.table('library_cache').select('shows').eq('device_id', device_id).execute()
        if not result.data:
            return {}

        shows = result.data[0].get('shows', [])
        mapping: Dict[int, int] = {}
        for show in shows:
            tmdb_raw = show.get('tmdbId') or show.get('tmdb_id')
            tvdb_raw = show.get('tvdbId') or show.get('tvdb_id')

            tmdb_id = _coerce_int(tmdb_raw)
            tvdb_id = _coerce_int(tvdb_raw)

            if tmdb_id is None or tvdb_id is None:
                continue

            mapping[tmdb_id] = tvdb_id

        return mapping
    except Exception as exc:
        print(f"⚠️  Failed to load TV show mapping from cache: {exc}")
        return {}


def add_to_stage(operation: str, items: List[Dict], device_id: str = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add verified items to staging for bulk operations"""
    stage_id = str(uuid.uuid4())
    
    # Enrich items with fresh TMDB data
    processed_items = []
    library_show_map: Optional[Dict[int, int]] = None

    for item in items:
        # Just need the basics from AI
        tmdb_id = item.get("tmdb_id")
        media_type = item.get("media_type", "movie")
        reason = item.get("reason")  # Optional reason for explore recommendations
        source = item.get("source")
        skip_tmdb = (
            source == "library"
            or item.get("skip_tmdb") is True
            or item.get("skip_enrichment") is True
        )

        if not tmdb_id:
            print(f"⚠️ Skipping item without tmdb_id: {item}")
            continue

        # Track optional IDs the frontend may need (e.g., TVDB for Sonarr)
        tvdb_value = item.get("tvdb_id") or item.get("tvdbId")
        tvdb_int = _coerce_int(tvdb_value)

        if skip_tmdb and operation != "explore":
            year_val = _coerce_int(item.get("year")) or 0
            processed_item = {
                "tmdb_id": int(tmdb_id),
                "title": item.get("title", "Unknown"),
                "year": year_val,
                "media_type": media_type,
                "poster_path": item.get("poster_path", "") or "",
                "verified": item.get("verified", True),
                "tvdb_id": tvdb_int,
                "overview": (item.get("overview") or "").strip(),
            }
            if reason:
                processed_item["reason"] = reason
            processed_items.append(processed_item)
            print(f"⚡ Cached staging: {processed_item['title']} ({year_val}) without TMDB lookup")
            continue

        # Fetch full data from TMDB
        try:
            endpoint = "movie" if media_type == "movie" else "tv"
            tmdb_url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}"
            request_params = {"api_key": TMDB_API_KEY}
            if media_type == "tv":
                request_params["append_to_response"] = "external_ids"

            response = httpx.get(
                tmdb_url,
                params=request_params,
                timeout=5.0
            )

            if response.status_code == 200:
                tmdb_data = response.json()

                # Extract year properly
                year = 0
                if media_type == "movie" and tmdb_data.get("release_date"):
                    year = int(tmdb_data["release_date"][:4])
                elif media_type == "tv" and tmdb_data.get("first_air_date"):
                    year = int(tmdb_data["first_air_date"][:4])

                # Capture TVDB ID for shows if available
                if media_type == "tv" and not tvdb_value:
                    tvdb_value = tmdb_data.get("external_ids", {}).get("tvdb_id")

                tvdb_int = _coerce_int(tvdb_value)

                if media_type == "tv" and tvdb_int is None:
                    if library_show_map is None:
                        library_show_map = _load_library_show_map(device_id)
                    try:
                        tmdb_key = int(tmdb_id)
                    except (TypeError, ValueError):
                        tmdb_key = None
                    if tmdb_key is not None:
                        tvdb_int = library_show_map.get(tmdb_key)

                processed_item = {
                    "tmdb_id": int(tmdb_id),
                    "title": tmdb_data.get("title" if media_type == "movie" else "name", "Unknown"),
                    "year": year,
                    "media_type": media_type,
                    "poster_path": tmdb_data.get("poster_path", ""),
                    "verified": True,  # Already verified if we got TMDB data
                    "tvdb_id": tvdb_int,
                    "overview": (tmdb_data.get("overview") or "").strip(),
                }

                # Include reason if provided (for explore operations)
                if reason:
                    processed_item["reason"] = reason

                processed_items.append(processed_item)
                print(f"✅ Enriched: {processed_item['title']} ({processed_item['year']}) with poster: {processed_item['poster_path']}")
            else:
                print(f"❌ TMDB API error for {tmdb_id}: {response.status_code}")
                # Still add basic item even if TMDB fails
                basic_item = {
                    "tmdb_id": int(tmdb_id),
                    "title": item.get("title", "Unknown"),
                    "year": 0,
                    "media_type": media_type,
                    "poster_path": "",
                    "verified": False,
                    "tvdb_id": tvdb_int,
                    "overview": (item.get("overview") or "").strip(),
                }
                if reason:
                    basic_item["reason"] = reason
                processed_items.append(basic_item)
        except Exception as e:
            print(f"❌ Error fetching TMDB data for {tmdb_id}: {e}")
            # Still add basic item
            basic_item = {
                "tmdb_id": int(tmdb_id),
                "title": item.get("title", "Unknown"),
                "year": 0,
                "media_type": media_type,
                "poster_path": "",
                "verified": False,
                "tvdb_id": tvdb_int,
                "overview": (item.get("overview") or "").strip(),
            }
            if reason:
                basic_item["reason"] = reason
            processed_items.append(basic_item)

        # Fallback to cached mapping if enrichment didn't provide TVDB IDs
        if media_type == "tv" and tvdb_int is None and processed_items:
            if library_show_map is None:
                library_show_map = _load_library_show_map(device_id)
            try:
                tmdb_key = int(tmdb_id)
            except (TypeError, ValueError):
                tmdb_key = None
            if tmdb_key is not None:
                resolved_tvdb = library_show_map.get(tmdb_key)
                if resolved_tvdb is not None:
                    processed_items[-1]["tvdb_id"] = resolved_tvdb
                    tvdb_int = resolved_tvdb

    # Save to Supabase
    try:
        data = {
            "stage_id": stage_id,
            "operation": operation,
            "items": processed_items,
            "status": "pending",
            "device_id": device_id  # Store device_id instead of user_id
        }
        
        result = supabase.table("staged_operations").insert(data).execute()
        print(f"✅ Saved to Supabase: stage_id={stage_id} with {len(processed_items)} enriched items")
        
        return {
            "stage_id": stage_id,
            "operation": operation,  # Include operation type for frontend routing
            "staged_count": len(processed_items),
            "ready_for_ui": True,
            "items": processed_items,
            "saved": True,
            "message": f"Staged {len(processed_items)} items with stage_id: {stage_id}",
            "params": params or {}
        }
    except Exception as e:
        # If Supabase fails, still return the staging info
        print(f"Failed to save to Supabase: {e}")
        return {
            "stage_id": stage_id,
            "operation": operation,  # Include operation type for frontend routing
            "staged_count": len(processed_items),
            "ready_for_ui": True,
            "items": processed_items,
            "saved": False,
            "error": str(e),
            "params": params or {}
        }

# Unified system prompt for both explore and library management
UNIFIED_PROMPT = """You are Z, an AI assistant for media discovery and library management.

ZERO-KNOWLEDGE ARCHITECTURE:
- You have access to library_cache from Supabase (synced by device daily)
- You NEVER receive or use server credentials - complete privacy!
- Use get_library_stats, get_all_movies, get_all_shows to check what's in library
- Device executes ALL operations locally after you stage them

QUERY TYPE DETECTION:

EXPLORE QUERIES (recommendations, browsing):
- "show me movies like Pearl"
- "Leonardo DiCaprio movies"
- "sci-fi from the 90s"
- "what should I watch tonight"
→ Use add_to_stage(operation="explore", items=[...])
→ Chat response should be a brief, friendly acknowledgement. Vary your wording each time—don't repeat the same phrases. The UI automatically displays results, so there's no need to mention buttons, tapping, or any manual actions.
→ Examples: "Found some options for you.", "Here you go.", "Got a few picks.", "Ready when you are."
→ DO NOT list individual titles, summaries, reasons, stage IDs, or commands in chat for explore operations — the UI handles those.
→ You may mention how many options were staged, but nothing more detailed.
→ ALWAYS check library first with get_all_movies/get_all_shows and filter out what they own

LIBRARY MANAGEMENT (add/remove/update):
- "add Ocean's 11"
- "delete all movies under 5.0"
- "add all Christopher Nolan movies"
→ 1-3 items: add_to_queue(items=[...]) - instant execution
→ 4+ items: add_to_stage(operation="add/remove/update", items=[...]) - shows modal
→ When staging items that already exist in the library, include title/year/profile/poster info from the cache, set media_type, and set "source": "library" to skip redundant TMDB lookups.
→ Use get_radarr_quality_profiles / get_sonarr_quality_profiles to match user intent (e.g., “4K”, “1080p HDR”) to the correct profile id and name.
→ For update operations, include params with target_quality_profile_id/name (and search_after_update when useful) so the device can apply the right profile automatically.

WORKFLOWS:

ACTOR QUERIES ("Leonardo DiCaprio movies"):
1. search_person → get person_id
2. get_person_credits → complete filmography
3. get_all_movies → filter out owned titles
4. Filter by year/rating as requested
5. Build items array with reasons: [{{"tmdb_id": 577922, "media_type": "movie", "reason": "direct companion/prequel by the same director; same aesthetic and cast"}}, ...]
6. add_to_stage(operation="explore", items=<array>)

DIRECTOR QUERIES ("Christopher Nolan movies"):
1. search_person → get person_id
2. get_person_credits → filter to movies (exclude TV)
3. get_all_movies → filter out owned titles
4. Build items array with reasons
5. add_to_stage(operation="explore", items=<array>)

THEME QUERIES ("sci-fi from the 90s"):
1. web_search for comprehensive lists
2. Verify each title via TMDB search to get IDs
3. get_all_movies/get_all_shows → filter out owned titles
4. Build items array with reasons
5. add_to_stage(operation="explore", items=<array>)

TMDB SEARCH RULES:
- When calling search_movies/search_shows, use simple queries: just the title, optionally the 4-digit year.
- Do NOT include director names, actor names, or words like "film"/"movie" in the query.
- Prefer format: "Title 2019" or just "Title". The backend will use the year as a hint when present.
- If a query returns 0 results, try at most one alternate phrasing. Avoid spamming many similar calls.

LIBRARY OPERATIONS:

Add 1-3 items:
→ search_movies → add_to_queue(items=[{tmdb_id: 161, media_type: "movie"}])
→ Say: "I've added Ocean's 11 to your library"

Add 4+ items:
→ search_movies (12x) → add_to_stage(operation="add", items=[...12 movies...])
→ Say: "I've staged 12 Nolan movies for your review"

Delete items:
→ get_all_movies → filter → add_to_stage(operation="remove", items=[...])
→ Say: "I've staged X movies for removal"

RESPONSE STYLES:

Explore (operation="explore"):
- Chat copy: short acknowledgement only (no title lists, no reasons, no stage IDs). Let the button speak for itself.
- Staged items MUST still include a concise "reason" field per item (1-2 sentences max) so the Explore UI can render it.

Library Management (operation="add/remove/update"):
- Queue (1-3): "I've added X to your library"
- Stage (4+): "I've staged X items for review"
- Concise and action-oriented
- No need for "reason" field in items for library operations

OPERATIONS:
- "explore" = recommendations/browsing
- "add" = green badge (adding to library)
- "remove" = red badge (deleting from library)
- "update" = blue badge (modifying library items)

TV SHOWS:
- Default to Season 1 unless specified

OUTPUT:
- Always include items parameter in add_to_stage
- Return stage_id when staging operations"""


# Unified tool definitions for both explore and library management
def get_unified_tools():
    """Returns unified toolset for both explore and library management"""
    return [
        {"type": "web_search"},
        {
            "type": "function",
            "name": "get_library_stats",
            "description": "Get statistics about the user's media library (movie and show counts)",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_all_movies",
            "description": "Get all movies in user's library with TMDB IDs. Use to check what they already have or for bulk operations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_all_shows",
            "description": "Get all TV shows in user's library. Use to check what they already have or for bulk operations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "search_movies_in_library",
            "description": "Search for movies already in the user's library",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Movie title to search for"}
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "search_movies",
            "description": "Search for movies on TMDB. Returns director information and full details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Movie title to search for"}
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "search_shows",
            "description": "Search for TV shows on TMDB",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "TV show title to search for"}
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "search_person",
            "description": "Search for people (actors, directors, etc.) on TMDB to get their ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Person's name to search for"}
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_person_credits",
            "description": "Get all movies and TV shows for a person by their TMDB ID. Returns complete filmography with ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer", "description": "TMDB person ID from search_person"}
                },
                "required": ["person_id"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_show_episodes",
            "description": "Get detailed episode list for a TV show including which episodes you have. Requests episode data from device in real-time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_title": {"type": "string", "description": "Title of the TV show to get episodes for"}
                },
                "required": ["show_title"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_radarr_quality_profiles",
            "description": "Return cached Radarr quality profiles (id & name) for selecting target quality levels.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_sonarr_quality_profiles",
            "description": "Return cached Sonarr quality profiles (id & name) for selecting target quality levels.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "add_to_queue",
            "description": "Queue 1-3 items for instant execution on device. Device auto-executes without showing modal. Use for quick adds to library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "Array of 1-3 items with tmdb_id and media_type",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tmdb_id": {"type": "integer"},
                                "media_type": {"type": "string", "enum": ["movie", "tv"]}
                            },
                            "required": ["tmdb_id", "media_type"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "add_to_stage",
            "description": "Stage items for visual display or bulk operations. Use 'explore' for poster-style recommendations, 'add' for adding to library, 'remove' for deletion, 'update' for modifications. MUST include items array.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["explore", "add", "remove", "update", "queue"],
                        "description": "Operation type: explore (poster mosaic), add (green), remove (red), update (blue), queue (instant)"
                    },
                    "items": {
                        "type": "array",
                        "description": "Array of items with tmdb_id, media_type, and optional reason (for explore operations)",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "tmdb_id": {"type": "integer"},
                                "media_type": {"type": "string", "enum": ["movie", "tv"]},
                                "reason": {
                                    "type": ["string", "null"],
                                    "description": "Why this item is recommended (for explore operations only)"
                                },
                                "title": {
                                    "type": ["string", "null"],
                                    "description": "Known title from the user's library cache"
                                },
                                "year": {
                                    "type": ["integer", "null"],
                                    "description": "Release year from the user's library cache"
                                },
                                "tvdb_id": {
                                    "type": ["integer", "null"],
                                    "description": "TVDB identifier (required for Sonarr operations on shows)"
                                },
                                "poster_path": {
                                    "type": ["string", "null"],
                                    "description": "Optional poster path or URL when available"
                                },
                                "source": {
                                    "type": ["string", "null"],
                                    "description": "Set to \"library\" to reuse cached metadata and skip TMDB lookups"
                                },
                                "verified": {
                                    "type": ["boolean", "null"],
                                    "description": "True if the item metadata came from the user's library cache"
                                }
                            },
                            "required": ["tmdb_id", "media_type"],
                            "additionalProperties": False
                        }
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional operation parameters (e.g., target_quality_profile_id/name for update flows)",
                        "additionalProperties": True
                    }
                },
                "required": ["operation", "items"],
                "additionalProperties": False
            },
            "strict": False
        }
    ]


# Function dispatcher
def execute_function(function_name: str, arguments: dict, device_id: str) -> dict:
    """Execute a function call and return the result - ZERO-KNOWLEDGE!"""
    try:
        # Discover tools
        if function_name == "search_movies":
            return search_movies(arguments["query"])
        elif function_name == "search_shows":
            return search_shows(arguments["query"])
        elif function_name == "search_person":
            return search_person(arguments["query"])
        elif function_name == "get_person_credits":
            return get_person_credits(arguments["person_id"])
        elif function_name == "add_to_stage":
            return add_to_stage(
                arguments["operation"],
                arguments["items"],
                device_id,
                arguments.get("params"),
            )
        # Chat tools - all use library_cache, never call user servers!
        elif function_name == "get_library_stats":
            return get_library_stats(device_id)
        elif function_name == "get_all_movies":
            return get_all_movies(device_id)
        elif function_name == "get_all_shows":
            return get_all_shows(device_id)
        elif function_name == "get_show_episodes":
            return get_show_episodes(arguments["show_title"], device_id)
        elif function_name == "search_movies_in_library":
            return search_movies_in_library(arguments["query"], device_id)
        elif function_name == "get_radarr_quality_profiles":
            return get_radarr_quality_profiles(device_id)
        elif function_name == "get_sonarr_quality_profiles":
            return get_sonarr_quality_profiles(device_id)
        elif function_name == "add_to_queue":
            return add_to_queue(arguments["items"], device_id)
        else:
            return {"error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"error": str(e)}

class DeviceRegisterRequest(BaseModel):
    device_id: str
    hmac_key: str
    receipt_token: str  # RevenueCat purchase token for verification
    user_id: Optional[str] = None  # Supabase user ID for tier-based rate limiting

@app.post("/device/register")
async def register_device(request: DeviceRegisterRequest):
    """Register a new device with its HMAC key - verifies Mega subscription via RevenueCat"""
    try:
        # Validate UUID format
        device_uuid = uuid.UUID(request.device_id)
        device_id = str(device_uuid)

        # Verify Mega subscription with RevenueCat
        logger.info(f"🎫 Verifying RevenueCat subscription for device {device_id[:8]}...")

        if not REVENUECAT_SECRET_KEY:
            logger.error("RevenueCat secret key not configured!")
            raise HTTPException(status_code=500, detail="Subscription verification unavailable")

        if not request.receipt_token:
            raise HTTPException(status_code=403, detail="Receipt token required")

        # DEVELOPMENT MODE: Skip verification for anonymous IDs (simulator/testing)
        if request.receipt_token.startswith("$RCAnonymousID:"):
            logger.warning(f"⚠️  DEV MODE: Bypassing RevenueCat verification for anonymous user")
            # Allow registration for testing
        else:
            # Verify with RevenueCat API
            try:
                # Get subscriber info from RevenueCat
                rc_response = httpx.get(
                    f"https://api.revenuecat.com/v1/subscribers/{request.receipt_token}",
                    headers={
                        "Authorization": f"Bearer {REVENUECAT_SECRET_KEY}",
                        "Content-Type": "application/json"
                    },
                    timeout=10.0
                )

                if rc_response.status_code != 200:
                    logger.warning(f"RevenueCat verification failed: {rc_response.status_code}")
                    raise HTTPException(status_code=403, detail="Invalid subscription")

                # Check for active Mega entitlement
                subscriber_data = rc_response.json()
                entitlements = subscriber_data.get("subscriber", {}).get("entitlements", {})

                # Check if user has active Mega entitlement
                mega_entitlement = entitlements.get("Mega", {})
                is_mega_active = mega_entitlement.get("expires_date") is not None

                # Also check if it's not expired
                if is_mega_active:
                    expires_date_str = mega_entitlement.get("expires_date")
                    expires_date = datetime.fromisoformat(expires_date_str.replace("Z", "+00:00"))
                    is_mega_active = expires_date > datetime.now(expires_date.tzinfo)

                if not is_mega_active:
                    logger.warning(f"No active Mega subscription for {request.receipt_token}")
                    raise HTTPException(status_code=403, detail="Mega subscription required")

                logger.info(f"✅ Verified Mega subscription for device {device_id[:8]}")

            except httpx.RequestError as e:
                logger.error(f"RevenueCat API error: {e}")
                raise HTTPException(status_code=502, detail="Failed to verify subscription")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unexpected error verifying subscription: {e}")
                raise HTTPException(status_code=500, detail="Subscription verification failed")

        # Check if device already exists
        existing = supabase.table('device_keys').select('device_id').eq('device_id', device_id).execute()

        if existing.data:
            # Update existing device's HMAC key and user_id (if provided)
            update_data = {
                'hmac_key': request.hmac_key,
                'last_used': datetime.now().isoformat()
            }
            if request.user_id:
                update_data['user_id'] = request.user_id

            supabase.table('device_keys').update(update_data).eq('device_id', device_id).execute()

            logger.info(f"🔄 Updated HMAC key for device {device_id[:8]}...")
        else:
            # Register new device
            insert_data = {
                'device_id': device_id,
                'hmac_key': request.hmac_key,
                'created_at': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat(),
                'request_count': 0
            }
            if request.user_id:
                insert_data['user_id'] = request.user_id

            supabase.table('device_keys').insert(insert_data).execute()

            logger.info(f"✅ Registered new device {device_id[:8]}...")

        return {"status": "registered", "device_id": device_id}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid device ID format")
    except Exception as e:
        logger.error(f"Device registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/chat")
async def chat(
    request: ChatRequest,
    device_auth: tuple[str, str] = Depends(verify_device_subscription)
):
    """Unified chat endpoint - handles both explore and library management"""
    device_id, hmac_key = device_auth
    await check_rate_limit(device_id)

    try:
        print(f"💬 CHAT: {request.message} (device: {device_id[:8]}...)")
        print(f"📦 ZERO-KNOWLEDGE: Backend will NEVER receive or use server credentials!")

        input_messages = [{"role": "user", "content": request.message}]
        tools = get_unified_tools()
        max_iterations = 25
        iteration = 0
        pending_commands = []  # Track commands to send to device
        stage_id = None  # Track if any tool returned a stage_id
        operation_type = None  # Track the operation type (explore, add, remove, etc.)
        agent_start_time = time.time()

        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")

            if time.time() - agent_start_time > AGENT_TIMEOUT_SECONDS:
                raise HTTPException(
                    status_code=504,
                    detail=f"Agent timed out after {AGENT_TIMEOUT_SECONDS} seconds"
                )

            response = openai_client.responses.create(
                model="gpt-5-mini",
                instructions=UNIFIED_PROMPT,
                input=input_messages,
                tools=tools,
            )

            has_function_calls = False
            for item in response.output:
                if item.type == "function_call":
                    has_function_calls = True
                    function_name = item.name
                    arguments = json.loads(item.arguments)
                    call_id = item.call_id

                    print(f"  🔧 {function_name}({arguments})")

                    result = execute_function(function_name, arguments, device_id)

                    # Check if result contains a stage_id (from add_to_queue or add_to_stage)
                    if isinstance(result, dict) and result.get('stage_id'):
                        stage_id = result.get('stage_id')
                        # Track operation type to determine response format
                        if result.get('operation'):
                            operation_type = result.get('operation')
                        print(f"  → Got stage_id: {stage_id}, operation: {operation_type}")

                    # Check if result requires library sync
                    if isinstance(result, dict) and result.get('requires_sync'):
                        pending_commands.append({
                            "action": "sync_library",
                            "services": ["radarr", "sonarr"]
                        })
                        print("  → Added sync_library command")

                    input_messages.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": function_name,
                        "arguments": item.arguments
                    })

                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result)
                    })

                    print(f"  ✅ {json.dumps(result)[:100]}...")

            if not has_function_calls:
                output_text = ""
                for item in response.output:
                    if item.type == "message":
                        for content in item.content:
                            if hasattr(content, 'text'):
                                output_text = content.text
                                break
                        break

                # Check if this is a staged operation (stage_id returned)
                # Stage IDs are UUIDs, so if response looks like one, treat as stage
                if len(output_text) == 36 and output_text.count('-') == 4:
                    print(f"📦 Staged operation: {output_text}")
                    response_data = {"response": output_text, "stage_id": output_text, "staged": True}
                    if operation_type:
                        response_data["operation"] = operation_type
                        print(f"📦 Including operation '{operation_type}' for direct stage response")
                    if pending_commands:
                        response_data["commands"] = pending_commands
                    return response_data

                print(f"💬 Response: {output_text[:100]}...")
                response_data = {"response": output_text}

                # If any tool returned a stage_id, include it in response along with operation type
                if stage_id:
                    response_data["staged"] = True
                    response_data["stage_id"] = stage_id
                    if operation_type:
                        response_data["operation"] = operation_type
                        print(f"📦 Including stage_id {stage_id} with operation '{operation_type}' in response")
                    else:
                        print(f"📦 Including stage_id in response: {stage_id}")

                if pending_commands:
                    response_data["commands"] = pending_commands
                    print(f"📤 Sending {len(pending_commands)} commands to device")
                return response_data

        raise HTTPException(
            status_code=504,
            detail=f"Agent timed out after {AGENT_TIMEOUT_SECONDS} seconds"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat failed")

@app.get("/")
async def root():
    return {"status": "Z Assistant with Tools Running 🚀"}

@app.get("/health")
async def health():
    """Health check endpoint with Redis status"""
    redis_status = "connected" if redis_client is not None else "disabled"
    return {
        "status": "healthy",
        "redis": redis_status,
        "rate_limiting": "enabled" if redis_client is not None else "disabled"
    }

@app.get("/staged/{stage_id}")
async def get_staged_operation(stage_id: str):
    """Retrieve a staged operation by ID"""
    try:
        result = supabase.table("staged_operations").select("*").eq("stage_id", stage_id).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]
        else:
            raise HTTPException(status_code=404, detail="Staged operation not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
