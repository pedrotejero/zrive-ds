import requests
import time
import random

def api_request(url: str, params: dict, max_retries: int = 3, base_backoff: float = 1.0) -> dict:
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