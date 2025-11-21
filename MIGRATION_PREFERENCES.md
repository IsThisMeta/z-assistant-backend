# Database Migration: Add User Preferences

This migration adds support for user preferences to customize UI visibility (hide tabs).

## SQL Migration

Run this SQL in your Supabase database:

```sql
-- Add preferences column to device_keys table
ALTER TABLE device_keys
ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{"hide_modules_tab": false, "hide_calendar_tab": false}'::jsonb;

-- Create index for faster preference lookups (optional but recommended)
CREATE INDEX IF NOT EXISTS idx_device_keys_preferences ON device_keys USING GIN (preferences);
```

## Preferences Schema

The `preferences` column stores a JSON object with the following structure:

```json
{
  "hide_modules_tab": false,
  "hide_calendar_tab": false
}
```

## API Endpoints

### Get Preferences
```
GET /preferences/{device_id}
```

Returns the user preferences for the specified device.

**Response:**
```json
{
  "hide_modules_tab": false,
  "hide_calendar_tab": false
}
```

### Update Preferences
```
POST /preferences/{device_id}
```

Updates user preferences. Only provided fields will be updated.

**Request Body:**
```json
{
  "hide_modules_tab": true,
  "hide_calendar_tab": false
}
```

**Response:**
```json
{
  "status": "success",
  "preferences": {
    "hide_modules_tab": true,
    "hide_calendar_tab": false
  }
}
```

## Authentication

Both endpoints require device authentication via the `verify_device_subscription` dependency. Include the standard HMAC authentication headers.

## Notes

- Preferences are stored per device, not per user
- If preferences don't exist, defaults to `false` for all options
- Partial updates are supported (you can update just one preference at a time)
