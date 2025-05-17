import requests
import time
import random
import pandas as pd
import matplotlib.pyplot as plt


def api_request(
    url: str, params: dict, max_retries: int = 3, base_backoff: float = 1.0
) -> dict:
    """
    Generic API call with exponential backoff, jitter, and error handling.
    """
    for attempt in range(1, max_retries + 1):
        response = requests.get(url, params=params)

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                raise RuntimeError("Invalid JSON in response")

        elif response.status_code == 400:
            # Bad request, likely invalid parameters
            # open-meteo API provides an specific error message
            raise RuntimeError(f"Bad request: {response.reason}")

        elif response.status_code in (429, 502, 503, 504):
            # Handle Retry-After if provided
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                wait_time = float(retry_after)
            else:
                # Exponential backoff with jitter
                wait_time = base_backoff * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.5 * wait_time)
                wait_time += jitter

            if attempt == max_retries:
                break  # After the last attempt, don't sleep

            time.sleep(wait_time)

        else:
            response.raise_for_status()

    raise RuntimeError(f"Failed after {max_retries} attempts")


def check_date_format(date: str) -> str:
    """
    Date format Required by Open-Meteo API: ISO8601 (YYYY-MM-DD)
    """
    try:
        pd.to_datetime(date, format="%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date}. Expected format: YYYY-MM-DD")
    return date


def df_temporary_reduction(
    df: pd.DataFrame, aggregation_map: dict, freq: str = "ME"
) -> pd.DataFrame:
    """
    Resamples the data at given frequency (e.g., 'M' for monthly)
    and returns the aggregated DataFrame.
    """
    resampled = df.resample(freq).agg(aggregation_map)
    return resampled


def plot_variable(data: dict, variable: str):
    """
    Plots a given variable for multiple cities over time.
    """
    plt.figure(figsize=(10, 6))
    for city, df in data.items():
        plt.plot(df.index, df[variable], label=city)

    min_date = data[city].index.min().date()
    max_date = data[city].index.max().date()
    plt.title(f"{variable} evolution ({min_date} to {max_date})")
    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.legend()
    plt.tight_layout()
    plt.show()
