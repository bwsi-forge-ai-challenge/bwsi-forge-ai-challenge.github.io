"""Importing this package registers every backend. Add your own module here once it exists."""

from . import toy  # noqa: F401

try:  # the real backends need torch; toy mode works without it
    from . import hf  # noqa: F401
except ImportError as e:
    print(f"[telephone] Hugging Face backends unavailable ({e}); toy backends only.")
