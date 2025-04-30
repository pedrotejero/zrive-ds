import pytest
import pandas as pd
import requests
import src.module_1.aux_functions as aux_functions
from src.module_1.aux_functions import api_request, check_date_format, df_temporary_reduction

# Dummy response class to simulate requests.Response
class DummyResponse:
    def __init__(self, status_code, json_data=None, reason='', headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.reason = reason
        self.headers = headers or {}

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} Error")


# Tests for api_request
def test_api_request_success(monkeypatch):
    resp = DummyResponse(200, {'key': 'value'})
    monkeypatch.setattr(aux_functions.requests, 'get', lambda url, params: resp)

    result = api_request("http://example.com", {'p': 1})
    assert result == {'key': 'value'}


def test_api_request_invalid_json(monkeypatch):
    resp = DummyResponse(200, ValueError("no json"))
    monkeypatch.setattr(aux_functions.requests, 'get', lambda url, params: resp)

    with pytest.raises(RuntimeError) as excinfo:
        api_request("http://example.com", {})
    assert "Invalid JSON in response" in str(excinfo.value)


def test_api_request_bad_request(monkeypatch):
    resp = DummyResponse(400, None, reason="Bad params")
    monkeypatch.setattr(aux_functions.requests, 'get', lambda url, params: resp)

    with pytest.raises(RuntimeError) as excinfo:
        api_request("http://example.com", {})
    assert "Bad request: Bad params" in str(excinfo.value)


def test_api_request_retry_then_success(monkeypatch):
    calls = []
    def fake_get(url, params):
        calls.append(1)
        if len(calls) == 1:
            return DummyResponse(503, None, headers={'Retry-After': '0.1'})
        else:
            return DummyResponse(200, {'ok': True})

    monkeypatch.setattr(aux_functions.requests, 'get', fake_get)
    monkeypatch.setattr(aux_functions.time, 'sleep', lambda x: None)

    result = api_request("http://example.com", {})
    assert result == {'ok': True}
    assert len(calls) == 2


def test_api_request_retry_exhausted(monkeypatch):
    def fake_get(url, params):
        return DummyResponse(503, None)

    sleeps = []
    monkeypatch.setattr(aux_functions.requests, 'get', fake_get)
    monkeypatch.setattr(aux_functions.time, 'sleep', lambda x: sleeps.append(x))
    monkeypatch.setattr(aux_functions.random, 'uniform', lambda a, b: 0)

    with pytest.raises(RuntimeError) as excinfo:
        api_request("http://example.com", {}, max_retries=3, base_backoff=1.0)
    assert "Failed after 3 attempts" in str(excinfo.value)

    # Should have slept for attempts 1 and 2
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(1.0)
    assert sleeps[1] == pytest.approx(2.0)


# Tests for check_date_format
def test_check_date_format_valid():
    date_str = "2025-04-29"
    assert check_date_format(date_str) == date_str


def test_check_date_format_invalid():
    with pytest.raises(ValueError) as excinfo:
        check_date_format("29-04-2025")
    assert "Invalid date format: 29-04-2025" in str(excinfo.value)


# Tests for df_temporary_reduction
def test_df_temporary_reduction_monthly_sum():
    idx = pd.date_range(start='2025-01-01', periods=4, freq='ME')
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]}, index=idx)
    result = df_temporary_reduction(df, {'a': 'sum', 'b': 'sum'}, freq='ME')
    pd.testing.assert_frame_equal(result, df)
