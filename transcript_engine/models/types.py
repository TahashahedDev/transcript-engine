from collections.abc import Callable

# (progress: 0.0–1.0, message: human-readable stage description)
ProgressCallback = Callable[[float, str], None]
