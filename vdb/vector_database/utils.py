import subprocess
import weaviate.classes as wvc


def refresh_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error refreshing token: {result.stderr}")
        return None
    return result.stdout.strip()


def make_properties(fields):
    return [
        wvc.config.Property(name=name, data_type=data_type)
        for name, data_type in fields
    ]
