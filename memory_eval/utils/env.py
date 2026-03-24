"""Environment loading utilities."""

from dotenv import find_dotenv, load_dotenv


_ENV_LOADED = False


def load_project_env(*, force: bool = False) -> None:
    """Load variables from a local .env file without overriding existing env vars."""
    global _ENV_LOADED

    if _ENV_LOADED and not force:
        print("Environment already loaded, skipping. Set force=True to reload.")
        return

    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    load_dotenv(dotenv_path=dotenv_path or None, override=False)
    _ENV_LOADED = True