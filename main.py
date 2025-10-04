from fastapi import FastAPI, HTTPException, Header, Depends
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

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase client (for auth - will use user's token for RLS)
supabase: Client = create_client(
    os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Only for admin operations
)

# Get API keys from environment
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "eaba5719606a782018d06df21c4fe459")
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

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 40  # requests per 3 hours (like ChatGPT)
RATE_LIMIT_WINDOW = 10800  # 3 hours in seconds

class ChatRequest(BaseModel):
    message: str
    servers: dict  # Required - contains radarr and sonarr configs
    context: Optional[str] = None  # Optional context like "discover"

    @validator('servers')
    def validate_servers(cls, v):
        """Validate server configuration"""
        if not isinstance(v, dict):
            raise ValueError('servers must be a dictionary')

        # Validate URLs if radarr/sonarr exist
        for service in ['radarr', 'sonarr']:
            if service in v:
                if not isinstance(v[service], dict):
                    raise ValueError(f'{service} must be a dictionary')

                # Validate URL format
                url = v[service].get('url', '')
                if url and not re.match(r'^https?://', url):
                    raise ValueError(f'{service} URL must start with http:// or https://')

                # Validate API key exists
                if 'api_key' not in v[service]:
                    raise ValueError(f'{service} must have an api_key')

        return v

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

async def check_rate_limit(user_id: str):
    """Check and enforce rate limits per user using Upstash Redis"""
    if not redis_client:
        # Redis not available - skip rate limiting (dev mode)
        print(f"⚠️  Rate limiting skipped for {user_id} (Redis not configured)")
        return

    # Use Redis sorted set for time-based rate limiting
    # Key format: ratelimit:{user_id}
    # Score: timestamp
    # Value: unique request ID
    key = f"ratelimit:{user_id}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    try:
        # Remove old entries outside the time window
        redis_client.zremrangebyscore(key, 0, window_start)

        # Count requests in current window
        count = redis_client.zcard(key)

        # Check if limit exceeded
        if count >= RATE_LIMIT_REQUESTS:
            # Get the oldest request timestamp
            oldest = redis_client.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = int(oldest_time + RATE_LIMIT_WINDOW - now)

                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)}
                )

        # Add current request with timestamp as score
        request_id = f"{now}:{uuid.uuid4()}"
        redis_client.zadd(key, {request_id: now})

        # Set expiry on the key (cleanup after window expires)
        redis_client.expire(key, RATE_LIMIT_WINDOW)

        print(f"✅ Rate limit check passed for {user_id}: {count + 1}/{RATE_LIMIT_REQUESTS}")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Redis error - log but don't block request
        print(f"⚠️  Rate limit check failed for {user_id}: {e}")
        # Continue without rate limiting on Redis errors

def validate_url(url: str) -> bool:
    """Validate URL format"""
    return bool(re.match(r'^https?://[^\s]+$', url))

# Define tools that accept servers parameter
def delete_movie(movie_id: int, delete_files: bool, servers: dict) -> Dict[str, Any]:
    """Delete a movie from Radarr"""
    print(f"🔧 TOOL CALLED: delete_movie - Removing movie ID {movie_id}")
    
    try:
        # First get movie details to show what we're deleting
        movie_response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/movie/{movie_id}",
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        
        if movie_response.status_code != 200:
            return {"error": f"Movie with ID {movie_id} not found"}
        
        movie_data = movie_response.json()
        movie_title = movie_data.get("title", "Unknown")
        
        print(f"  → Deleting: {movie_title}")
        
        # Delete the movie
        delete_response = httpx.delete(
            f"{servers['radarr']['url']}/api/v3/movie/{movie_id}",
            params={"deleteFiles": str(delete_files).lower()},
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        
        if delete_response.status_code in [200, 202, 204]:
            print(f"  ✓ Successfully deleted {movie_title}")
            return {
                "success": True,
                "message": f"Deleted {movie_title}" + (" and its files" if delete_files else " (kept files)"),
                "deleted_movie": movie_title
            }
        else:
            return {"error": f"Failed to delete movie: {delete_response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

# Function for Responses API
def search_movies_in_library(query: str, servers: dict) -> Dict[str, Any]:
    """Search for movies already in your library"""
    print(f"🔧 TOOL CALLED: search_movies_in_library - Query: {query}")
    
    try:
        response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/movie",
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        
        if response.status_code != 200:
            return {"error": "Failed to get library"}
        
        all_movies = response.json()
        query_lower = query.lower()
        
        # Search for matches
        matches = []
        for movie in all_movies:
            if query_lower in movie.get("title", "").lower():
                matches.append({
                    "id": movie["id"],
                    "title": movie["title"],
                    "year": movie.get("year"),
                    "hasFile": movie.get("hasFile", False)
                })
        
        if not matches:
            return {"matches": [], "message": f"No movies found matching '{query}'"}
        
        print(f"  ✓ Found {len(matches)} matches")
        return {"matches": matches[:5]}  # Limit to 5 results
        
    except Exception as e:
        return {"error": str(e)}

# Function for Responses API
def search_media(query: str, servers: dict) -> Dict[str, Any]:
    """Search for movies or TV shows"""
    results = {"movies": [], "shows": []}
    
    # Search Radarr
    try:
        response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/movie/lookup",
            params={"term": query},
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        if response.status_code == 200:
            movies = response.json()[:3]  # Limit to 3
            results["movies"] = [
                {
                    "title": m.get("title"), 
                    "year": m.get("year"), 
                    "tmdbId": m.get("tmdbId"),
                    "in_library": m.get("id") is not None
                }
                for m in movies
            ]
    except Exception as e:
        results["movie_error"] = str(e)
    
    return results

# Function for Responses API
def get_all_shows(servers: dict) -> Dict[str, Any]:
    """Get list of all TV shows in the library with genres"""
    print("🔧 TOOL CALLED: get_all_shows")
    try:
        response = httpx.get(
            f"{servers['sonarr']['url']}/api/v3/series",
            headers={"X-Api-Key": servers['sonarr']['api_key']},
            timeout=10.0
        )
        if response.status_code == 200:
            shows = response.json()
            show_list = []
            for show in shows[:10]:  # Limit to 10 for readability
                show_list.append({
                    "title": show.get("title"),
                    "genres": show.get("genres", []),
                    "year": show.get("year")
                })
            print(f"  ✓ Retrieved {len(show_list)} shows")
            return {"shows": show_list, "total": len(shows)}
    except Exception as e:
        return {"error": str(e)}

# Function for Responses API
def get_library_stats(servers: dict) -> Dict[str, Any]:
    """Get real statistics about the user's media library"""
    print("🔧 TOOL CALLED: get_library_stats")
    stats = {"movies": 0, "shows": 0}
    
    try:
        print("  → Calling Radarr API...")
        # Get movie count from Radarr
        response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/movie",
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        if response.status_code == 200:
            movies = response.json()
            stats["movies"] = len(movies)
            print(f"  ✓ Found {len(movies)} movies")
    except Exception as e:
        stats["movie_error"] = str(e)
    
    try:
        print("  → Calling Sonarr API...")
        # Get show count from Sonarr
        response = httpx.get(
            f"{servers['sonarr']['url']}/api/v3/series",
            headers={"X-Api-Key": servers['sonarr']['api_key']},
            timeout=10.0
        )
        if response.status_code == 200:
            shows = response.json()
            stats["shows"] = len(shows)
            print(f"  ✓ Found {len(shows)} shows")
    except Exception as e:
        stats["show_error"] = str(e)
    
    return stats

# Function for Responses API
def add_movie_to_radarr(tmdb_id: int, title: str, servers: dict) -> Dict[str, Any]:
    """Add a movie to Radarr by TMDB ID"""
    # Convert to int if it comes as float
    tmdb_id = int(tmdb_id)
    print(f"🔧 TOOL CALLED: add_movie_to_radarr - Adding {title} (TMDB: {tmdb_id})")
    
    try:
        # First, let's try a general lookup with the TMDB ID
        lookup_response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/movie/lookup",
            params={"term": f"tmdb:{tmdb_id}"},
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        
        print(f"  → Lookup URL: {servers['radarr']['url']}/api/v3/movie/lookup?term=tmdb:{tmdb_id}")
        print(f"  → Response status: {lookup_response.status_code}")
        
        if lookup_response.status_code != 200:
            print(f"  → Response body: {lookup_response.text}")
            return {"error": f"Failed to lookup movie: {lookup_response.status_code}"}
        
        movies = lookup_response.json()
        if not movies:
            return {"error": "Movie not found"}
        
        movie_data = movies[0]  # Get the first result
        
        # Check if already in library
        if movie_data.get("id"):
            return {"error": f"{title} is already in your library!"}
        
        # Get root folders to use the first available one
        folders_response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/rootfolder",
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        
        root_folder = "/movies"  # default
        if folders_response.status_code == 200:
            folders = folders_response.json()
            if folders:
                root_folder = folders[0]["path"]
                print(f"  → Using root folder: {root_folder}")
        
        # Get quality profiles to use the first available one
        profiles_response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/qualityprofile",
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        
        quality_profile_id = 1  # default
        if profiles_response.status_code == 200:
            profiles = profiles_response.json()
            if profiles:
                quality_profile_id = profiles[0]["id"]
                print(f"  → Using quality profile: {profiles[0]['name']} (ID: {quality_profile_id})")
        
        # Add to Radarr
        add_data = {
            "title": movie_data["title"],
            "tmdbId": tmdb_id,
            "year": movie_data["year"],
            "qualityProfileId": quality_profile_id,
            "rootFolderPath": root_folder,
            "monitored": True,
            "addOptions": {
                "searchForMovie": True
            }
        }
        
        add_response = httpx.post(
            f"{servers['radarr']['url']}/api/v3/movie",
            json=add_data,
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        
        if add_response.status_code in [200, 201]:
            print(f"  ✓ Successfully added {title} to Radarr!")
            return {"success": True, "message": f"Added {title} to your library and started searching for it!"}
        else:
            print(f"  → Add failed: {add_response.status_code}")
            print(f"  → Response: {add_response.text}")
            return {"error": f"Failed to add movie: {add_response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

# Function for Responses API
def get_all_movies(servers: dict) -> Dict[str, Any]:
    """Get list of all movies in the library with genres"""
    print("🔧 TOOL CALLED: get_all_movies")
    try:
        response = httpx.get(
            f"{servers['radarr']['url']}/api/v3/movie",
            headers={"X-Api-Key": servers['radarr']['api_key']},
            timeout=10.0
        )
        if response.status_code == 200:
            movies = response.json()
            movie_list = []
            for movie in movies[:20]:  # Limit to 20 for readability
                movie_list.append({
                    "title": movie.get("title"),
                    "year": movie.get("year"),
                    "genres": movie.get("genres", []),
                    "tmdbId": movie.get("tmdbId"),
                    "sizeOnDisk": movie.get("sizeOnDisk", 0) / (1024**3),  # Convert to GB
                    "hasFile": movie.get("hasFile", False)
                })
            print(f"  ✓ Retrieved {len(movie_list)} movies (showing first 20 of {len(movies)} total)")
            return {"movies": movie_list, "total": len(movies)}
    except Exception as e:
        return {"error": str(e)}

# Function for Responses API
def search_movies(query: str) -> Dict[str, Any]:
    """Search for movies on TMDB and return a list of results"""
    print(f"🔧 TOOL CALLED: search_movies - Query: {query}")
    
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": query
        }
        
        response = httpx.get(search_url, params=params, timeout=5.0)
        if response.status_code == 200:
            results = response.json().get("results", [])
            
            # Return top 5 results with relevant info
            movies = []
            for item in results[:5]:
                year = None
                if item.get("release_date"):
                    try:
                        year = int(item["release_date"][:4])
                    except:
                        pass
                
                # Get director info
                director = None
                try:
                    credits_response = httpx.get(
                        f"https://api.themoviedb.org/3/movie/{item['id']}/credits",
                        params={"api_key": TMDB_API_KEY},
                        timeout=3.0
                    )
                    if credits_response.status_code == 200:
                        credits = credits_response.json()
                        directors = [crew["name"] for crew in credits.get("crew", []) if crew.get("job") == "Director"]
                        if directors:
                            director = directors[0]
                except:
                    pass

                movies.append({
                    "tmdb_id": item["id"],
                    "title": item["title"],
                    "year": year,
                    "director": director,
                    "poster_path": item.get("poster_path"),
                    "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else ""
                })
            
            print(f"  ✓ Found {len(movies)} movie results")
            return {"movies": movies, "total_found": len(results)}
        
        return {"movies": [], "error": f"API error: {response.status_code}"}
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
def add_to_stage(operation: str, items: List[Dict], device_id: str = None) -> Dict[str, Any]:
    """Add verified items to staging for bulk operations"""
    stage_id = str(uuid.uuid4())
    
    # Enrich items with fresh TMDB data
    processed_items = []

    for item in items:
        # Just need the basics from AI
        tmdb_id = item.get("tmdb_id")
        media_type = item.get("media_type", "movie")
        
        if not tmdb_id:
            print(f"⚠️ Skipping item without tmdb_id: {item}")
            continue
            
        # Fetch full data from TMDB
        try:
            endpoint = "movie" if media_type == "movie" else "tv"
            tmdb_url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}"
            
            response = httpx.get(
                tmdb_url,
                params={"api_key": TMDB_API_KEY},
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
                
                processed_item = {
                    "tmdb_id": int(tmdb_id),
                    "title": tmdb_data.get("title" if media_type == "movie" else "name", "Unknown"),
                    "year": year,
                    "media_type": media_type,
                    "poster_path": tmdb_data.get("poster_path", ""),
                    "verified": True  # Already verified if we got TMDB data
                }
                processed_items.append(processed_item)
                print(f"✅ Enriched: {processed_item['title']} ({processed_item['year']}) with poster: {processed_item['poster_path']}")
            else:
                print(f"❌ TMDB API error for {tmdb_id}: {response.status_code}")
                # Still add basic item even if TMDB fails
                processed_items.append({
                    "tmdb_id": int(tmdb_id),
                    "title": item.get("title", "Unknown"),
                    "year": 0,
                    "media_type": media_type,
                    "poster_path": "",
                    "verified": False
                })
        except Exception as e:
            print(f"❌ Error fetching TMDB data for {tmdb_id}: {e}")
            # Still add basic item
            processed_items.append({
                "tmdb_id": int(tmdb_id),
                "title": item.get("title", "Unknown"),
                "year": 0,
                "media_type": media_type,
                "poster_path": "",
                "verified": False
            })
    
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
            "staged_count": len(processed_items),
            "operation": operation,
            "ready_for_ui": True,
            "items": processed_items,
            "saved": True,
            "message": f"Staged {len(processed_items)} items with stage_id: {stage_id}"
        }
    except Exception as e:
        # If Supabase fails, still return the staging info
        print(f"Failed to save to Supabase: {e}")
        return {
            "stage_id": stage_id,
            "staged_count": len(processed_items),
            "operation": operation,
            "ready_for_ui": True,
            "items": processed_items,
            "saved": False,
            "error": str(e)
        }

# Helper function to create tools with servers injected
# System prompt for DISCOVER view
DISCOVER_PROMPT = """You are Z, a media discovery assistant. Find movies and TV shows, then stage them for display.

IMPORTANT: Be smart and decisive. When users ask for something obvious, don't overthink it:
- "The Matrix" = The Matrix (1999), not the sequels
- "Star Wars" = Star Wars: A New Hope (1977)
- "The Godfather" = The Godfather (1972), not Part II
- For any movie series, assume they mean the FIRST/ORIGINAL movie unless they specify otherwise

LIBRARY AWARENESS:
- You can check what's already in the user's library with get_all_movies and get_all_shows
- If user asks for things they "don't have" or "missing from library", check their library first
- Filter out titles they already own before staging results

WORKFLOW:

FOR ACTOR QUERIES (like "Leonardo DiCaprio movies" or "films with Tom Hanks"):
1. Use search_person to find the actor and get their TMDB person_id
2. Use get_person_credits with that person_id to get their COMPLETE filmography
3. Filter results by year, rating (vote_average), or other criteria as requested
4. If user wants items "not in library", call get_all_movies/get_all_shows to filter
5. Build items array from filtered results: items=[{{"tmdb_id": 577922, "media_type": "movie"}}, ...]
6. Call add_to_stage(operation="discover", items=<the array>)
7. Return ONLY the stage_id

FOR DIRECTOR QUERIES (like "Christopher Nolan movies"):
1. Search for 10-20 movies/shows using search_movies or search_shows tools
   - Search results include director information - USE IT to filter!
   - If user asks for "Christopher Nolan movies", ONLY include results where director="Christopher Nolan"
2. After ALL searches complete, manually build the items array from the results
3. Call add_to_stage(operation="discover", items=<the array>)
4. Return ONLY the stage_id

FOR OTHER QUERIES (like "sci-fi from the 90s"):
1. Use web_search to get comprehensive lists
2. Then search TMDB for each title to verify and get IDs
3. Build items array and call add_to_stage
4. Return ONLY the stage_id

YOU MUST PASS THE ITEMS PARAMETER - NEVER call add_to_stage with only operation!
ALWAYS return ONLY the stage_id, nothing else."""

# System prompt for CHAT assistant
CHAT_PROMPT = """You are Z, a friendly media library assistant. Help users manage their Radarr and Sonarr libraries.

THRESHOLD RULE - CRITICAL:
- 1-3 items: Execute directly using add_movie_to_radarr, delete_movie, etc.
- 4+ items: MUST use add_to_stage for user confirmation
  - operation="add" for adding media (green badge in UI)
  - operation="remove" for deleting media (red badge in UI)
  - operation="update" for modifying media (blue badge in UI)

EXAMPLES:
User: "Add The Matrix"
→ Direct: Use add_movie_to_radarr (1 item)

User: "Add Inception, Tenet, and Interstellar"
→ Direct: Use add_movie_to_radarr for each (3 items)

User: "Add all Christopher Nolan movies"
→ Stage: Search for all titles → call add_to_stage(operation="add", items=[...]) → return stage_id

User: "Delete all movies under 5.0 rating"
→ Stage: Get library → filter → call add_to_stage(operation="remove", items=[...]) → return stage_id

CAPABILITIES:
- Check library statistics
- Search library for specific titles
- Add/delete/update movies in Radarr
- Answer questions about media collection

For TV shows:
- Always add Season 1 by default unless user specifies otherwise

When staging (4+ items):
1. Search/filter to get the items
2. Build items array with tmdb_id and media_type
3. Call add_to_stage with appropriate operation
4. Return ONLY the stage_id

Never make up data - always use tools to get real information."""


# Tool definitions for DISCOVER view
def get_discover_tools():
    """Returns tools for discover/search functionality"""
    return [
        {"type": "web_search"},
        {
            "type": "function",
            "name": "search_movies",
            "description": "Search for movies on TMDB. Returns director information for filtering.",
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
            "description": "Search for TV shows on TMDB.",
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
            "description": "Search for people (actors, directors, etc.) on TMDB to get their ID.",
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
            "name": "get_all_movies",
            "description": "Get all movies in user's library to filter out what they already have",
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
            "description": "Get all TV shows in user's library to filter out what they already have",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "add_to_stage",
            "description": "Stage items for display in UI. MUST include items array.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "Operation type like 'discover'"},
                    "items": {
                        "type": "array",
                        "description": "Array of items with tmdb_id and media_type",
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
                "required": ["operation", "items"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]

# Tool definitions for CHAT assistant
def get_chat_tools():
    """Returns tools for library management chat"""
    return [
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
            "description": "Get all movies in user's library with TMDB IDs for bulk operations",
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
            "description": "Get all TV shows in user's library for bulk operations",
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
            "name": "add_movie_to_radarr",
            "description": "Add a single movie to Radarr (for 1-3 items only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "tmdb_id": {"type": "integer", "description": "TMDB ID of the movie"},
                    "title": {"type": "string", "description": "Title of the movie"}
                },
                "required": ["tmdb_id", "title"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "delete_movie",
            "description": "Delete a single movie from Radarr (for 1-3 items only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {"type": "integer", "description": "Radarr movie ID"},
                    "delete_files": {"type": "boolean", "description": "Whether to delete files"}
                },
                "required": ["movie_id", "delete_files"],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "search_movies",
            "description": "Search for movies on TMDB to get their ID",
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
            "description": "Search for TV shows on TMDB to get their ID",
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
            "name": "add_to_stage",
            "description": "Stage 4+ items for user confirmation. MUST include items array.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "remove", "update"], "description": "Operation type: add (green), remove (red), update (blue)"},
                    "items": {
                        "type": "array",
                        "description": "Array of items with tmdb_id and media_type",
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
                "required": ["operation", "items"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]


# Function dispatcher
def execute_function(function_name: str, arguments: dict, servers: dict, device_id: str = None) -> dict:
    """Execute a function call and return the result"""
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
            return add_to_stage(arguments["operation"], arguments["items"], device_id)
        # Chat tools
        elif function_name == "get_library_stats":
            return get_library_stats(servers)
        elif function_name == "get_all_movies":
            return get_all_movies(servers)
        elif function_name == "get_all_shows":
            return get_all_shows(servers)
        elif function_name == "search_movies_in_library":
            return search_movies_in_library(arguments["query"], servers)
        elif function_name == "add_movie_to_radarr":
            return add_movie_to_radarr(arguments["tmdb_id"], arguments["title"], servers)
        elif function_name == "delete_movie":
            return delete_movie(arguments["movie_id"], arguments["delete_files"], servers)
        else:
            return {"error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/discover")
async def discover(
    request: ChatRequest,
    device_auth: tuple[str, str] = Depends(verify_device_subscription)
):
    """Discover endpoint - search for media and return stage_id"""
    device_id, hmac_key = device_auth
    await check_rate_limit(device_id)

    try:
        print(f"🔍 DISCOVER: {request.message} (device: {device_id[:8]}...)")

        # Decrypt credentials if encrypted - CRITICAL FIX!
        servers = request.servers
        if servers and isinstance(next(iter(servers.values()), None), str):
            # Credentials are encrypted - decrypt them
            servers = decrypt_credentials(servers, hmac_key)
            print(f"🔐 Decrypted credentials for {list(servers.keys())}")

        input_messages = [{"role": "user", "content": request.message}]
        tools = get_discover_tools()
        max_iterations = 25
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")

            response = openai_client.responses.create(
                model="gpt-5-mini",
                instructions=DISCOVER_PROMPT,
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

                    result = execute_function(function_name, arguments, servers, device_id)  # Use decrypted servers!

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

                stage_id = output_text.strip()
                print(f"🎯 Returning stage_id: {stage_id}")
                return {"response": stage_id, "stage_id": stage_id}

        raise HTTPException(status_code=500, detail="Request took too long")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Discover error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

class DeviceRegisterRequest(BaseModel):
    device_id: str
    hmac_key: str
    receipt_token: str  # RevenueCat purchase token for verification

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
            # Update existing device's HMAC key
            supabase.table('device_keys').update({
                'hmac_key': request.hmac_key,
                'last_used': datetime.now().isoformat()
            }).eq('device_id', device_id).execute()

            logger.info(f"🔄 Updated HMAC key for device {device_id[:8]}...")
        else:
            # Register new device
            supabase.table('device_keys').insert({
                'device_id': device_id,
                'hmac_key': request.hmac_key,
                'created_at': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat(),
                'request_count': 0
            }).execute()

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
    """Chat endpoint - conversational library assistant with HMAC encryption"""
    device_id, hmac_key = device_auth
    await check_rate_limit(device_id)

    try:
        print(f"💬 CHAT: {request.message} (device: {device_id[:8]}...)")

        # Decrypt credentials if encrypted
        servers = request.servers
        if servers and isinstance(next(iter(servers.values()), None), str):
            # Credentials are encrypted - decrypt them
            servers = decrypt_credentials(servers, hmac_key)
            print(f"🔐 Decrypted credentials for {list(servers.keys())}")

        input_messages = [{"role": "user", "content": request.message}]
        tools = get_chat_tools()
        max_iterations = 25
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")

            response = openai_client.responses.create(
                model="gpt-5-mini",
                instructions=CHAT_PROMPT,
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

                    result = execute_function(function_name, arguments, servers, device_id)

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
                    return {"response": output_text, "stage_id": output_text, "staged": True}

                print(f"💬 Response: {output_text[:100]}...")
                return {"response": output_text}

        raise HTTPException(status_code=500, detail="Request took too long")

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
