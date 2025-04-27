RESPONSE_SCHEMA = {
  "type": "object",
  "properties": {
    "latitude": {
      "type": "number"
    },
    "longitude": {
      "type": "number"
    },
    "generationtime_ms": {
      "type": "number"
    },
    "utc_offset_seconds": {
      "type": "integer"
    },
    "timezone": {
      "type": "string"
    },
    "timezone_abbreviation": {
      "type": "string"
    },
    "elevation": {
      "type": "number"
    },
    "daily_units": {
      "type": "object",
      "properties": {
        "time": {
          "type": "string"
        },
        "temperature_2m_mean": {
          "type": "string"
        },
        "precipitation_sum": {
          "type": "string"
        },
        "wind_speed_10m_max": {
          "type": "string"
        }
      }
    },
    "daily": {
      "type": "object",
      "properties": {
        "time": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "temperature_2m_mean": {
          "type": "array",
          "items": {
            "type": "number"
          }
        },
        "precipitation_sum": {
          "type": "array",
          "items": {
            "type": "number"
          }
        },
        "wind_speed_10m_max": {
          "type": "array",
          "items": {
            "type": "number"
          }
        }
      }
    }
  }
}