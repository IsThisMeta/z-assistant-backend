from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import httpx
import os
from typing import Dict, Any, List
import uuid
import json
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role key to bypass RLS
)

class ChatRequest(BaseModel):
    message: str
    servers: dict  # Required - contains radarr and sonarr configs
    context: str = None  # Optional context like "discover"

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
            "api_key": "eaba5719606a782018d06df21c4fe459",
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
                        params={"api_key": "eaba5719606a782018d06df21c4fe459"},
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
            "api_key": "eaba5719606a782018d06df21c4fe459",
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
            "api_key": "eaba5719606a782018d06df21c4fe459",
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
            "api_key": "eaba5719606a782018d06df21c4fe459"
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
def add_to_stage(operation: str, items: List[Dict], user_id: str = None) -> Dict[str, Any]:
    """Add verified items to staging for bulk operations"""
    stage_id = str(uuid.uuid4())
    
    # Enrich items with fresh TMDB data
    processed_items = []
    tmdb_api_key = "eaba5719606a782018d06df21c4fe459"
    
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
                params={"api_key": tmdb_api_key},
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
            "user_id": user_id
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
# System prompt for the assistant
SYSTEM_PROMPT = """You are Z, a media server assistant. Use tools to get real data.

IMPORTANT: Be smart and decisive. When users ask for something obvious, don't overthink it:
- "The Matrix" = The Matrix (1999), not the sequels
- "Star Wars" = Star Wars: A New Hope (1977)
- "The Godfather" = The Godfather (1972), not Part II
- For any movie series, assume they mean the FIRST/ORIGINAL movie unless they specify otherwise

COMPLEX REQUESTS:

For ACTOR queries (like "Leonardo DiCaprio movies" or "films with Tom Hanks"):
1. Use search_person to find the actor and get their TMDB person_id
2. Use get_person_credits with that person_id to get their COMPLETE filmography
3. Filter by year, rating (vote_average), or other criteria as requested
4. This gives you VERIFIED, COMPLETE results directly from TMDB

For DIRECTOR queries (like "Christopher Nolan movies"):
1. Search for individual movie titles you know (Following, Memento, etc.)
2. Use director information from search results to verify
3. Christopher Nolan has: Following (1998), Memento (2000), Insomnia (2002), Batman Begins (2005), The Dark Knight (2008), The Prestige (2006), Inception (2010), The Dark Knight Rises (2012), Interstellar (2014), Dunkirk (2017), Tenet (2020), Oppenheimer (2023)
4. Be COMPREHENSIVE - find ALL their works, not just 5-6

For OTHER queries (like "sci-fi from the 90s"):
1. Use web_search to get comprehensive lists
2. Then search TMDB for each title to verify and get IDs

For TV shows:
- Always add just Season 1 (pilot season) by default unless the user specifies otherwise
- "Add The Office" = Add Season 1 only
- "Add all of The Office" or "Add The Office complete series" = Add all seasons

SEARCHING AND STAGING:
When users want to find media:
1. Use search_movies or search_shows to find options
2. Pick the most relevant results from the search
3. Use add_to_stage to stage them for the UI
4. For ambiguous searches, include multiple relevant results

DISCOVER VIEW REQUESTS - CRITICAL WORKFLOW:
If the message contains "[CONTEXT: DISCOVER VIEW]":

FOR ACTOR QUERIES (like "movies with Leonardo DiCaprio in 2025"):
1. Use search_person("Leonardo DiCaprio") to get person_id
2. Use get_person_credits(person_id) to get ALL their movies/shows
3. Filter results by year, rating, or other criteria
4. Build items array from filtered results: items=[{{"tmdb_id": 577922, "media_type": "movie"}}, ...]
5. Call add_to_stage(operation="discover", items=<the array>)
6. Return ONLY the stage_id

FOR DIRECTOR/TITLE QUERIES (like "Christopher Nolan movies"):
1. Search for 10-20 movies/shows using search_movies or search_shows tools
   - Search results include director information - USE IT to filter!
   - If user asks for "Christopher Nolan movies", ONLY include results where director="Christopher Nolan"
   - Don't just match by title - verify the director matches what user requested
2. After ALL searches complete, manually build the items array from the results:
   - Take the tmdb_id from EACH search result that matches the criteria
   - Create this exact format: items=[{{"tmdb_id": 577922, "media_type": "movie"}}, {{"tmdb_id": 614911, "media_type": "movie"}}, ...]
3. Call add_to_stage(operation="discover", items=<the array you built>)
4. Return ONLY the stage_id

EXAMPLE - If search_movies("Tenet") returned tmdb_id: 577922, director: "Christopher Nolan":
  Call: add_to_stage(operation="discover", items=[{{"tmdb_id": 577922, "media_type": "movie"}}])

YOU MUST PASS THE ITEMS PARAMETER - NEVER call add_to_stage with only operation!

When asked to delete media:
1. First search for it in the library to get the ID
2. Delete with files by default unless user says otherwise
3. Report what was deleted

Never make up data - always use tools to get real information. But DO use your knowledge to be thorough!"""


# Define tools in Responses API format
def get_tool_definitions():
    """Returns tool definitions in Responses API format"""
    return [
        {"type": "web_search"},  # Built-in web search tool
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


# Function dispatcher
def execute_function(function_name: str, arguments: dict, servers: dict) -> dict:
    """Execute a function call and return the result"""
    try:
        if function_name == "search_movies":
            return search_movies(arguments["query"])
        elif function_name == "search_shows":
            return search_shows(arguments["query"])
        elif function_name == "search_person":
            return search_person(arguments["query"])
        elif function_name == "get_person_credits":
            return get_person_credits(arguments["person_id"])
        elif function_name == "add_to_stage":
            return add_to_stage(arguments["operation"], arguments["items"])
        else:
            return {"error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Prepare input message
        input_message = request.message
        if request.context == "discover":
            input_message = f"[CONTEXT: DISCOVER VIEW] {request.message}"
            print(f"🎨 Discover context detected - AI will stage results for visual display")

        # Prepare instructions with system prompt
        input_messages = [
            {"role": "user", "content": input_message}
        ]

        # Get tool definitions
        tools = get_tool_definitions()

        # Agent loop - call responses.create until no more tool calls
        max_iterations = 25
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")

            # Call Responses API
            response = openai_client.responses.create(
                model="gpt-5-mini",
                instructions=SYSTEM_PROMPT,
                input=input_messages,
                tools=tools,
            )

            # Check for function calls in output
            has_function_calls = False
            for item in response.output:
                if item.type == "function_call":
                    has_function_calls = True
                    function_name = item.name
                    arguments = json.loads(item.arguments)
                    call_id = item.call_id

                    print(f"  🔧 Calling: {function_name} with {arguments}")

                    # Execute the function
                    result = execute_function(function_name, arguments, request.servers)

                    # Append function call to history
                    input_messages.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": function_name,
                        "arguments": item.arguments
                    })

                    # Append function result
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result)
                    })

                    print(f"  ✅ Result: {json.dumps(result)[:100]}...")

            # If no function calls, we have the final response
            if not has_function_calls:
                # Extract text from message output
                output_text = ""
                for item in response.output:
                    if item.type == "message":
                        for content in item.content:
                            if hasattr(content, 'text'):
                                output_text = content.text
                                break
                        break

                # For discover context, extract stage_id
                if request.context == "discover":
                    stage_id = output_text.strip()
                    print(f"🎯 Discover mode - stage_id: {stage_id}")
                    return {"response": stage_id, "stage_id": stage_id}

                return {"response": output_text}

        # Max iterations reached
        raise HTTPException(status_code=500, detail="Max iterations reached without completion")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "Z Assistant with Tools Running 🚀"}

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