from pathlib import Path
import sys
import os
import re
import logging, warnings
logging.captureWarnings(True)

def find_lumapi_path_recursive(start_dirs=None, max_depth=4):
    """
    Recursively search for 'v*/api/python' paths starting from given directories.
    Returns the first valid path found.
    """
    if start_dirs is None:
        if os.name == "nt":
            # Windows defaults
            start_dirs = [Path("C:/Program Files"), Path("C:/Program Files (x86)")]
        else:
            # macOS and Linux (example path, update if needed)
            start_dirs = [Path("/Applications"), Path("/opt")]

    lumapi_pattern = re.compile(r"v\d+.*[/\\]api[/\\]python", re.IGNORECASE)

    def within_depth(base: Path, current: Path, max_depth: int) -> bool:
        try:
            rel_parts = current.relative_to(base).parts
            return len(rel_parts) <= max_depth
        except ValueError:
            return False

    for base_dir in start_dirs:
        if not base_dir.exists():
            continue

        for path in base_dir.rglob("*"):
            if path.is_dir() and within_depth(base_dir, path, max_depth):
                if lumapi_pattern.search(str(path)):
                    return path.resolve()

    return None


def add_lumapi_to_path(lumapi_path=None):
    if any("lumapi" in str(p) for p in sys.path):
        return

    # Check environment variable
    env_path = os.getenv("LUMAPI_PATH")
    if env_path:
        env_path = Path(env_path).resolve()
        if env_path.exists():
            sys.path.append(str(env_path))
            return

    # Try to find lumapi path
    lumapi_path = lumapi_path or find_lumapi_path_recursive()
    if lumapi_path:
        sys.path.append(str(lumapi_path))
    else:
        warnings.warn(
            "Could not find 'lumapi'. Please specify lumapi path manually."
        )
