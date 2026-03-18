"""macOS Keychain management for API keys."""

import getpass
import sys

import keyring

SERVICE_NAME = "scotus-pipeline"
KEY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google (Gemini)",
}


def get_key(provider: str) -> str:
    """Read an API key from the macOS Keychain."""
    secret = keyring.get_password(SERVICE_NAME, provider)
    if not secret:
        print(f"ERROR: No API key found for '{provider}' in Keychain.")
        print("Please run 'python -m scotus setup' first.")
        sys.exit(1)
    return secret


def get_all_keys() -> dict[str, str]:
    """Read all API keys from the Keychain."""
    return {provider: get_key(provider) for provider in KEY_NAMES}


def verify_keys() -> bool:
    """Check whether all required keys are stored."""
    for provider in KEY_NAMES:
        if not keyring.get_password(SERVICE_NAME, provider):
            return False
    return True


def setup_keys() -> None:
    """Interactively prompt for API keys and store them in the Keychain."""
    print("SCOTUS Pipeline — API Key Setup")
    print("Keys are stored securely in the macOS Keychain.\n")

    for provider, label in KEY_NAMES.items():
        existing = keyring.get_password(SERVICE_NAME, provider)
        if existing:
            answer = input(f"  {label} key already exists. Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                print(f"  {label} key kept.\n")
                continue

        key = getpass.getpass(f"  Enter {label} API key: ").strip()
        if not key:
            print(f"  Skipped.\n")
            continue

        keyring.set_password(SERVICE_NAME, provider, key)
        print(f"  {label} key saved.\n")

    if verify_keys():
        print("All API keys are configured.")
    else:
        missing = [label for p, label in KEY_NAMES.items()
                    if not keyring.get_password(SERVICE_NAME, p)]
        print(f"WARNING: Missing keys: {', '.join(missing)}")
