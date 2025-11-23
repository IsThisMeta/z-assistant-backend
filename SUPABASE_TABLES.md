# Supabase Tables for On-Demand Episode Fetching

## New Tables Needed

### 1. `data_fetch_commands`
Stores commands for device to fetch episode data on-demand.

```sql
CREATE TABLE data_fetch_commands (
  request_id UUID PRIMARY KEY,
  device_id UUID NOT NULL,
  action TEXT NOT NULL, -- 'fetch_episodes'
  show_title TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'completed', 'failed'
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  completed_at TIMESTAMP WITH TIME ZONE,
  FOREIGN KEY (device_id) REFERENCES device_keys(device_id)
);

-- Index for device polling
CREATE INDEX idx_data_fetch_device_status ON data_fetch_commands(device_id, status);
```

### 2. `episode_cache`
Stores episode details fetched from device.

```sql
CREATE TABLE episode_cache (
  id BIGSERIAL PRIMARY KEY,
  device_id UUID NOT NULL,
  show_title TEXT NOT NULL,
  episodes JSONB NOT NULL, -- Array of episode objects
  synced_at TIMESTAMP WITH TIME ZONE NOT NULL,
  FOREIGN KEY (device_id) REFERENCES device_keys(device_id),
  UNIQUE(device_id, show_title)
);

-- Index for lookups
CREATE INDEX idx_episode_cache_device_show ON episode_cache(device_id, show_title);
```

## Episode Data Format

```json
{
  "episodes": [
    {
      "season_number": 1,
      "episode_number": 1,
      "title": "Pilot",
      "air_date": "2008-01-20",
      "has_file": true,
      "overview": "..."
    },
    {
      "season_number": 1,
      "episode_number": 2,
      "title": "Cat's in the Bag...",
      "air_date": "2008-01-27",
      "has_file": false,
      "overview": "..."
    }
  ]
}
```

## Flow

1. AI calls `get_show_episodes("Breaking Bad")`
2. Backend stages command in `data_fetch_commands` with status='pending'
3. Device polls for commands, finds the request
4. Device calls Sonarr episodes API
5. Device uploads episode data to `episode_cache`
6. Device updates command status='completed'
7. Backend (waiting with 3x5s retry) reads from `episode_cache`
8. Backend returns episode list to AI

## Zero-Knowledge Architecture

- Backend never receives server credentials
- Backend never calls Sonarr API
- Device does all the work
- Backend only orchestrates via Supabase

---

## Show Recommendations ("Up Next" Feature)

### 3. `show_recommendations_cache`
Stores AI-generated show recommendations for Mega/Ultra users. Similar to `deep_cuts_cache` but for TV shows.

```sql
CREATE TABLE IF NOT EXISTS show_recommendations_cache (
    device_id UUID PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
    generated_at TIMESTAMPTZ,
    next_generation_at TIMESTAMPTZ,
    is_generating BOOLEAN DEFAULT FALSE,
    generation_started_at TIMESTAMPTZ,
    generation_duration_ms INTEGER,
    prompt_version TEXT DEFAULT 'v1',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_show_recommendations_device_id
ON show_recommendations_cache(device_id);

CREATE INDEX IF NOT EXISTS idx_show_recommendations_next_gen
ON show_recommendations_cache(next_generation_at);

ALTER TABLE show_recommendations_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Devices can view own show recommendations"
ON show_recommendations_cache FOR SELECT
USING (device_id = current_setting('request.jwt.claims.device_id')::UUID);

CREATE POLICY "Devices can update own show recommendations"
ON show_recommendations_cache FOR UPDATE
USING (device_id = current_setting('request.jwt.claims.device_id')::UUID);

CREATE POLICY "Devices can insert own show recommendations"
ON show_recommendations_cache FOR INSERT
WITH CHECK (device_id = current_setting('request.jwt.claims.device_id')::UUID);
```

## Recommendation Data Format

```json
{
  "recommendations": [
    {
      "title": "The Bear",
      "year": 2022,
      "genres": ["Drama", "Comedy"],
      "reason": "Matches your love for character-driven dramas with strong ensemble casts",
      "popularity_score": 9,
      "seasons": 3,
      "tmdb_id": 136315,
      "poster_path": "/rS4f6MYLXyWjJi96kAGWVINmC96.jpg"
    }
  ]
}
```

## API Endpoints

- **POST** `/recommendations/up-next/generate` - Triggers weekly show recommendation generation
  - Requires: Device authentication + subscription verification (Mega/Ultra)
  - Headers: `X-Subscription-Tier: mega|ultra`
  - Returns: Generation status, recommendation count, duration

- **GET** `/recommendations/up-next` - Retrieves cached recommendations
  - Requires: Device authentication
  - Returns: Recommendations array, timestamps, generation status

## Generation Flow

1. User triggers generation (or weekly automated job)
2. Backend fetches library shows + watch history from cache
3. AI analyzes viewing patterns and generates 10-15 show recommendations
4. Filters out shows already in library (popularity_score >= 6)
5. Enriches with TMDB data (posters, IDs) using `/search/tv`
6. Stores in `show_recommendations_cache` with 7-day refresh cycle
7. Client displays recommendations with "Up Next" UI

---

## Magic Recommendations - Dynamic Themed Sections with 8-Week History

### 4-7. Magic Recommendation Tables

Four tables for AI-generated themed recommendations with dynamic section titles:
- `magic_movies_cache` - Dynamic themed movie sections
- `magic_movies_cast_crew_cache` - People-based movie sections
- `magic_shows_cache` - Dynamic themed TV show sections
- `magic_shows_cast_crew_cache` - People-based TV show sections

All tables use identical schema pattern:

```sql
CREATE TABLE IF NOT EXISTS <table_name> (
    device_id UUID PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    section_title TEXT NOT NULL,  -- AI-generated dynamic title
    section_theme TEXT,  -- Theme description (not used for cast/crew tables)
    featured_people JSONB,  -- Only for cast/crew tables
    recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
    history JSONB NOT NULL DEFAULT '[]'::jsonb,  -- 8-week title tracking
    generated_at TIMESTAMPTZ,
    next_generation_at TIMESTAMPTZ,
    is_generating BOOLEAN DEFAULT FALSE,
    generation_started_at TIMESTAMPTZ,
    generation_duration_ms INTEGER,
    prompt_version TEXT DEFAULT 'v1',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

See SQL section above for complete schemas with indexes and RLS policies.

## Magic Recommendations Data Format

### Magic Movies
```json
{
  "section_title": "Neon-Soaked Cyberpunk",
  "section_theme": "Tech-noir and cyberpunk films with stunning visual aesthetics",
  "recommendations": [
    {
      "title": "Ghost in the Shell",
      "year": 1995,
      "director": "Mamoru Oshii",
      "genres": ["Animation", "Sci-Fi", "Thriller"],
      "reason": "Pioneering cyberpunk anime with philosophical depth and gorgeous visuals",
      "relevance_score": 9,
      "tmdb_id": 9323,
      "poster_path": "/iGh...jpg"
    }
  ],
  "history": [
    {"title": "Ghost in the Shell", "added_at": "2025-01-15T10:00:00Z"},
    {"title": "Akira", "added_at": "2025-01-08T10:00:00Z"}
  ]
}
```

### Magic Movies Cast & Crew
```json
{
  "section_title": "The Fincher Universe",
  "featured_people": [
    {
      "name": "David Fincher",
      "role": "Director",
      "tmdb_id": 7467
    }
  ],
  "recommendations": [
    {
      "title": "Zodiac",
      "year": 2007,
      "director": "David Fincher",
      "genres": ["Crime", "Drama", "Mystery"],
      "reason": "Fincher's meticulous investigation thriller showcasing his signature precision",
      "relevance_score": 9,
      "featured_person": "David Fincher",
      "tmdb_id": 1949,
      "poster_path": "/yN..."
    }
  ],
  "history": [...]
}
```

### Magic Shows & Magic Shows Cast & Crew
Similar structure but for TV shows (includes `seasons` field, uses `/search/tv` endpoint).

## Magic Recommendations API Endpoints

**Movies - Themed:**
- **POST** `/recommendations/magic-movies/generate` - Generate dynamic themed movie section
- **GET** `/recommendations/magic-movies` - Retrieve cached themed movie recommendations

**Movies - Cast & Crew:**
- **POST** `/recommendations/magic-movies-cast-crew/generate` - Generate people-based movie section
- **GET** `/recommendations/magic-movies-cast-crew` - Retrieve cached people-based movie recommendations

**Shows - Themed:**
- **POST** `/recommendations/magic-shows/generate` - Generate dynamic themed show section
- **GET** `/recommendations/magic-shows` - Retrieve cached themed show recommendations

**Shows - Cast & Crew:**
- **POST** `/recommendations/magic-shows-cast-crew/generate` - Generate people-based show section
- **GET** `/recommendations/magic-shows-cast-crew` - Retrieve cached people-based show recommendations

All endpoints require:
- Device authentication + subscription verification (Mega/Ultra)
- Header: `X-Subscription-Tier: mega|ultra`

## Magic Recommendations Generation Flow

1. User triggers generation (or weekly automated job)
2. Backend fetches library + watch history + people cache
3. Backend retrieves 8-week history from cache to exclude previously recommended titles
4. AI analyzes patterns and **creates dynamic section title and theme**
   - For themed sections: Detects studio/style/genre patterns (e.g., "A24 Indie Darlings", "Prestige Crime Dramas")
   - For cast/crew sections: Identifies key people and builds section around them (e.g., "The Fincher Universe", "Ryan Murphy's World")
5. AI generates 8-12 recommendations matching the theme (relevance_score >= 6)
6. Filters out titles in library and in 8-week history
7. Enriches with TMDB data (posters, IDs)
8. Updates history: Adds new titles, removes entries >8 weeks old, caps at 100 items
9. Stores in cache with 7-day refresh cycle
10. Client displays with dynamic section title

## History Tracking (8-Week Loop Prevention)

The `history` field tracks all recommended titles for 8 weeks:

```json
{
  "history": [
    {"title": "Movie Title", "added_at": "2025-01-15T10:00:00Z"},
    {"title": "Another Movie", "added_at": "2025-01-08T10:00:00Z"},
    ...
  ]
}
```

- On each generation, new titles are added to history
- History is filtered to only include items from last 8 weeks
- Capped at 100 items maximum
- AI is instructed to avoid all titles in history
- Ensures recommendations stay fresh and don't loop
