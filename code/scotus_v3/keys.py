"""API key management.

Keys are read from environment variables first (OPENAI_API_KEY, ANTHROPIC_API_KEY,
GOOGLE_API_KEY). If a variable is not set, the macOS Keychain is used as a fallback
(service "scotus-pipeline"), which is how the original runs were configured.
"""

import getpass
import os
import sys

try:
    import keyring
except ImportError:  # keyring is optional on non-macOS machines
    keyring = None

ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

SERVICE_NAME = "scotus-pipeline"
KEY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google (Gemini)",
}


def get_key(provider: str) -> str:
    """Read an API key from the environment, falling back to the macOS Keychain."""
    secret = os.environ.get(ENV_VARS[provider])
    if not secret and keyring is not None:
        secret = keyring.get_password(SERVICE_NAME, provider)
    if not secret:
        print(f"ERROR: No API key for '{provider}'. Set {ENV_VARS[provider]} "
              f"or store it in the macOS Keychain (service '{SERVICE_NAME}').")
        sys.exit(1)
    return secret


def get_all_keys() -> dict[str, str]:
    """Read all API keys (environment first, then Keychain)."""
    return {provider: get_key(provider) for provider in KEY_NAMES}


def verify_keys() -> bool:
    """Check whether all required keys are stored."""
    for provider in KEY_NAMES:
        if os.environ.get(ENV_VARS[provider]):
            continue
        if keyring is None or not keyring.get_password(SERVICE_NAME, provider):
            return False
    return True


def setup_keys() -> None:
    """Interactively prompt for API keys and store them in the Keychain."""
    if keyring is None:
        print("keyring is not installed — set the environment variables instead:",
              ", ".join(ENV_VARS.values()))
        return
    print("SCOTUS Pipeline — API-Key Setup")
    print("Die Keys werden sicher im macOS Keychain gespeichert.\n")

    for provider, label in KEY_NAMES.items():
        existing = keyring.get_password(SERVICE_NAME, provider)
        if existing:
            answer = input(f"  {label} Key existiert bereits. Überschreiben? [y/N] ").strip().lower()
            if answer != "y":
                print(f"  {label} Key beibehalten.\n")
                continue

        key = getpass.getpass(f"  {label} API-Key eingeben: ").strip()
        if not key:
            print(f"  Übersprungen.\n")
            continue

        keyring.set_password(SERVICE_NAME, provider, key)
        print(f"  {label} Key gespeichert.\n")

    if verify_keys():
        print("Alle API-Keys sind konfiguriert.")
    else:
        missing = [label for p, label in KEY_NAMES.items()
                    if not keyring.get_password(SERVICE_NAME, p)]
        print(f"WARNUNG: Fehlende Keys: {', '.join(missing)}")
