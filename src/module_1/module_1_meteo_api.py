import pandas as pd
from jsonschema import validate, ValidationError

from static_variables import COORDINATES, VARIABLES, API_URL
from schema import RESPONSE_SCHEMA
from aux_functions import api_request, df_temporary_reduction, check_date_format


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
    start_date = '2010-01-01'
    end_date = '2020-12-31'

    start_date = check_date_format(start_date)
    end_date = check_date_format(end_date)
    if start_date > end_date:
        raise ValueError("Start date must be before end date.")

    processed_data = {}
    aggregation_map = {
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'sum',
        'wind_speed_10m_max': 'max'
    }
    for city in COORDINATES.keys():
        df_daily = get_data_meteo_api(city, start_date, end_date)
        df_monthly = df_temporary_reduction(df_daily, aggregation_map, freq='M')
        processed_data[city] = df_monthly

if __name__ == "__main__":
    main()
