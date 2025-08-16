import uuid
import subprocess
import pandas as pd
import weaviate.classes as wvc
from constants import NAMESPACE


def refresh_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error refreshing token: {result.stderr}")
        return None
    return result.stdout.strip()


def generate_uuid5(value):
    return uuid.uuid5(NAMESPACE, value)


def _row_exists(value):
    if pd.notna(value) and str(value).strip():
        return True


def _normalize_props(props: dict) -> dict:
    return {k: str(v).lower().strip() for k, v in props.items()}


def make_properties(fields):
    return [
        wvc.config.Property(name=name, data_type=data_type)
        for name, data_type, *_ in fields
    ]
