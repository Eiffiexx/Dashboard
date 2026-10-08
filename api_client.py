"""Secure, configurable data ingestion for the NilssonHedge dashboard.

The module deliberately keeps credentials out of source code. Configure the
connection with environment variables (or a local ``.env`` file) and never
commit real credentials or confidential data to Git.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from data_utils import clean_index_data


load_dotenv()


class ApiConfigurationError(ValueError):
    """Raised when API environment variables are incomplete or invalid."""


class ApiDataError(ValueError):
    """Raised when the API response cannot be converted to index data."""


@dataclass(frozen=True)
class ApiConfig:
    url: str
    auth_mode: str = "none"
    token: str = ""
    username: str = ""
    password: str = ""
    data_format: str = "auto"
    json_path: str = ""
    timeout_seconds: float = 30.0
    verify_ssl: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.url)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def api_config_from_env() -> ApiConfig:
    """Build API configuration from environment variables.

    Supported authentication modes:
    - ``none``: public endpoint
    - ``bearer``: ``Authorization: Bearer <token>``
    - ``basic``: username + password, including WordPress Application Passwords
    """

    try:
        timeout = float(os.getenv("NH_API_TIMEOUT_SECONDS", "30"))
    except ValueError as exc:
        raise ApiConfigurationError("NH_API_TIMEOUT_SECONDS must be numeric.") from exc

    return ApiConfig(
        url=os.getenv("NH_API_URL", "").strip(),
        auth_mode=os.getenv("NH_API_AUTH_MODE", "none").strip().lower(),
        token=os.getenv("NH_API_TOKEN", "").strip(),
        username=os.getenv("NH_API_USERNAME", "").strip(),
        password=os.getenv("NH_API_PASSWORD", "").strip(),
        data_format=os.getenv("NH_API_DATA_FORMAT", "auto").strip().lower(),
        json_path=os.getenv("NH_API_JSON_PATH", "").strip(),
        timeout_seconds=timeout,
        verify_ssl=_env_bool("NH_API_VERIFY_SSL", True),
    )


def validate_api_config(config: ApiConfig) -> None:
    if not config.url:
        raise ApiConfigurationError("NH_API_URL is not configured.")
    if not config.url.lower().startswith("https://"):
        raise ApiConfigurationError("NH_API_URL must use HTTPS when credentials are involved.")
    if config.auth_mode not in {"none", "bearer", "basic"}:
        raise ApiConfigurationError(
            "NH_API_AUTH_MODE must be one of: none, bearer, basic."
        )
    if config.auth_mode == "bearer" and not config.token:
        raise ApiConfigurationError("NH_API_TOKEN is required for bearer authentication.")
    if config.auth_mode == "basic" and (not config.username or not config.password):
        raise ApiConfigurationError(
            "NH_API_USERNAME and NH_API_PASSWORD are required for basic authentication."
        )
    if config.data_format not in {"auto", "csv", "json"}:
        raise ApiConfigurationError("NH_API_DATA_FORMAT must be auto, csv, or json.")
    if config.timeout_seconds <= 0:
        raise ApiConfigurationError("NH_API_TIMEOUT_SECONDS must be greater than zero.")


def _extract_json_path(payload: Any, path: str) -> Any:
    current = payload
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ApiDataError(f"NH_API_JSON_PATH could not find '{part}'.")
    return current


def _json_to_frame(payload: Any, json_path: str = "") -> pd.DataFrame:
    records = _extract_json_path(payload, json_path)

    if not json_path and isinstance(records, dict):
        for common_key in ("data", "records", "items", "results"):
            candidate = records.get(common_key)
            if isinstance(candidate, list):
                records = candidate
                break

    if isinstance(records, list):
        return pd.json_normalize(records)
    if isinstance(records, dict):
        try:
            return pd.DataFrame(records)
        except ValueError:
            return pd.DataFrame([records])
    raise ApiDataError("The JSON response must contain records as a list or object.")


def response_to_frame(
    response: requests.Response,
    *,
    data_format: str = "auto",
    json_path: str = "",
) -> pd.DataFrame:
    """Convert an HTTP response containing CSV or JSON into a DataFrame."""

    content_type = response.headers.get("Content-Type", "").lower()
    resolved_format = data_format
    if resolved_format == "auto":
        resolved_format = "json" if "json" in content_type else "csv"

    if resolved_format == "json":
        try:
            return _json_to_frame(response.json(), json_path)
        except requests.exceptions.JSONDecodeError as exc:
            raise ApiDataError("The API response is not valid JSON.") from exc

    try:
        return pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        raise ApiDataError("The API response is not valid CSV.") from exc


def fetch_index_data(
    config: ApiConfig | None = None,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download, validate, and standardize monthly index data from the API."""

    config = config or api_config_from_env()
    validate_api_config(config)

    headers = {
        "Accept": "application/json, text/csv;q=0.9, */*;q=0.8",
        "User-Agent": "NilssonHedge-Index-Dashboard/1.0",
    }
    auth = None
    if config.auth_mode == "bearer":
        headers["Authorization"] = f"Bearer {config.token}"
    elif config.auth_mode == "basic":
        auth = HTTPBasicAuth(config.username, config.password)

    client = session or requests.Session()
    response = client.get(
        config.url,
        headers=headers,
        auth=auth,
        timeout=config.timeout_seconds,
        verify=config.verify_ssl,
    )
    response.raise_for_status()

    raw = response_to_frame(
        response,
        data_format=config.data_format,
        json_path=config.json_path,
    )
    cleaned = clean_index_data(raw)
    if cleaned.empty:
        raise ApiDataError("The API returned no valid index rows.")
    return cleaned
