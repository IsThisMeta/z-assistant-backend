from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
import httpx
import os
from typing import Dict, Any, List
import uuid
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCPiEumwTgh8NqH2P5CRjzw78seF10QD_w"

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role key to bypass RLS
)

class ChatRequest(BaseModel):
    message: str
    servers: dict  # Required - contains radarr and sonarr configs

# Define tools that accept servers parameter
@tool
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

@tool
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

@tool
def search_media(query: str, servers: dict) -> Dict[str, Any]:
    """Search for movies or TV shows"""
    if "cats" in query.lower() and "2019" in query:
        return {"error": "🐱🚫 I'm allergic to that one"}
    
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

@tool
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

@tool
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

@tool
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

@tool
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

@tool
def tmdb_verify(title: str, year: int = None, media_type: str = "movie") -> Dict[str, Any]:
    """Quick TMDB verification for movies or TV shows - returns clean data"""
    try:
        # Determine search endpoint
        endpoint = "movie" if media_type == "movie" else "tv"
        search_url = f"https://api.themoviedb.org/3/search/{endpoint}"
        
        params = {
            "api_key": "eaba5719606a782018d06df21c4fe459",
            "query": title,
            "year": year if year else None
        }
        
        response = httpx.get(search_url, params=params, timeout=5.0)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                item = results[0]  # Best match
                
                # Different fields for movies vs shows
                if media_type == "movie":
                    return {
                        "tmdb_id": item["id"],
                        "title": item["title"],
                        "year": int(item.get("release_date", "0")[:4]) if item.get("release_date") else None,
                        "poster_path": item.get("poster_path"),
                        "media_type": "movie",
                        "verified": True
                    }
                else:  # TV show
                    return {
                        "tmdb_id": item["id"],
                        "title": item["name"],
                        "year": int(item.get("first_air_date", "0")[:4]) if item.get("first_air_date") else None,
                        "poster_path": item.get("poster_path"),
                        "media_type": "tv",
                        "verified": True
                    }
        
        return {"verified": False, "error": "Not found on TMDB"}
    except Exception as e:
        return {"verified": False, "error": str(e)}

@tool
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
            "saved": True
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
def create_tools_with_servers(servers: dict):
    """Create tool instances with servers pre-configured"""
    
    # Create wrapped versions of each tool
    @tool
    def get_library_stats_wrapped() -> Dict[str, Any]:
        """Get real statistics about the user's media library"""
        return get_library_stats.func(servers=servers)
    
    @tool
    def search_media_wrapped(query: str) -> Dict[str, Any]:
        """Search for movies or TV shows"""
        return search_media.func(query=query, servers=servers)
    
    @tool
    def get_all_shows_wrapped() -> Dict[str, Any]:
        """Get list of all TV shows in the library with genres"""
        return get_all_shows.func(servers=servers)
    
    @tool
    def get_all_movies_wrapped() -> Dict[str, Any]:
        """Get list of all movies in the library with genres"""
        return get_all_movies.func(servers=servers)
    
    @tool
    def add_movie_to_radarr_wrapped(tmdb_id: int, title: str) -> Dict[str, Any]:
        """Add a movie to Radarr by TMDB ID"""
        return add_movie_to_radarr.func(tmdb_id=tmdb_id, title=title, servers=servers)

    @tool
    def delete_movie_wrapped(movie_id: int, delete_files: bool = True) -> Dict[str, Any]:
        """Delete a movie from Radarr"""
        return delete_movie.func(movie_id=movie_id, delete_files=delete_files, servers=servers)

    @tool
    def search_movies_in_library_wrapped(query: str) -> Dict[str, Any]:
        """Search for movies already in your library"""
        return search_movies_in_library.func(query=query, servers=servers)
    
    @tool
    def tmdb_verify_wrapped(title: str, year: int = None, media_type: str = "movie") -> Dict[str, Any]:
        """Quick TMDB verification for movies or TV shows - returns clean data"""
        return tmdb_verify.func(title=title, year=year, media_type=media_type)
    
    @tool
    def add_to_stage_wrapped(operation: str, items: List[Dict]) -> Dict[str, Any]:
        """Add verified items to staging for bulk operations"""
        return add_to_stage.func(operation=operation, items=items)
    
    return [
        get_library_stats_wrapped,
        search_media_wrapped,
        get_all_shows_wrapped,
        get_all_movies_wrapped,
        add_movie_to_radarr_wrapped,
        delete_movie_wrapped,
        search_movies_in_library_wrapped,
        tmdb_verify_wrapped,
        add_to_stage_wrapped
    ]

# Create the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    temperature=0.3,
    max_retries=2,
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Z, a media server assistant. Use tools to get real data.

IMPORTANT: Be smart and decisive. When users ask for something obvious, don't overthink it:
- "The Matrix" = The Matrix (1999), not the sequels
- "Star Wars" = Star Wars: A New Hope (1977)
- "The Godfather" = The Godfather (1972), not Part II
- For any movie series, assume they mean the FIRST/ORIGINAL movie unless they specify otherwise

For TV shows:
- Always add just Season 1 (pilot season) by default unless the user specifies otherwise
- "Add The Office" = Add Season 1 only
- "Add all of The Office" or "Add The Office complete series" = Add all seasons
- This gives users a chance to try shows without committing to 10 seasons

Only ask for clarification when there's TRUE ambiguity (like two completely different movies with the same name).

When asked for recommendations:
1. First check what shows the user has
2. Analyze their taste based on genres/themes
3. Recommend a movie that fits their taste
4. Search for it to get the TMDB ID
5. Add it to their library

When asked to delete media:
1. First search for it in the library to get the ID
2. Confirm what you're about to delete only if there is ambiguity
3. Delete with files by default unless user says otherwise
4. Report what was deleted

When working with multiple items:
1. Use tmdb_verify to check each item
2. Use add_to_stage to prepare bulk operations
3. Report what will be staged for the user

Never make up data - always use tools to get real information."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Create tools with the user's servers
        user_tools = create_tools_with_servers(request.servers)
        
        # Create agent with user's tools
        agent = create_tool_calling_agent(llm, user_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=user_tools, verbose=True)
        
        # Execute the request
        result = agent_executor.invoke({
            "input": request.message
        })
        
        return {"response": result["output"]}
    except Exception as e:
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