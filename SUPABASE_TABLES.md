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
