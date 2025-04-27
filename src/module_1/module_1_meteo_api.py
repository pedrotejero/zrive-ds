import pandas as pd
from jsonschema import validate, ValidationError

from static_variables import COORDINATES, VARIABLES, API_URL
from schema import RESPONSE_SCHEMA
from aux_functions import api_request


def get_data_meteo_api(city: str, start_date: str = "2010-01-01", end_date: str = "2020-12-31") -> pd.DataFrame:
    """
    Fetches historical daily data for a city using the Open-Meteo API.
    """
    if city not in COORDINATES.keys():
        raise ValueError(f"Invalid city. Available cities: {', '.join(COORDINATES.keys())}")
    coords = COORDINATES[city]
    params = {
        "latitude": coords['latitude'],
        "longitude": coords['longitude'],
        "start_date": start_date,
        "end_date": end_date,
        "daily": VARIABLES,
    }
    response = api_request(API_URL, params)
    
    # Schema validation
    try:
        validate(instance=response, schema=RESPONSE_SCHEMA)
    except ValidationError as e:
        raise RuntimeError(f"Schema validation error: {e}")

    daily = response['daily']
    df = pd.DataFrame(daily)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


def main():
    get_data_meteo_api("London", "2010-01-01", "2020-12-31")

if __name__ == "__main__":
    main()
