import warnings
import functools
import csv
import os
from datetime import datetime
from inspect import signature, Parameter


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.get(*args)
        return dotdict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _read_header(path: str) -> list[str] | None:
    """Return the header row of an existing CSV, or None if unreadable/empty."""
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            return next(csv.reader(f))
    except (StopIteration, IOError):
        return None


def _build_schema(sig, extra_kwarg_keys: list[str]) -> list[str]:
    """
    Build the expected fieldnames list.

    Regular parameters (positional, keyword, VAR_POSITIONAL) are kept as-is.
    The VAR_KEYWORD (**kwargs) parameter is *removed* and replaced by the
    individual keys that were actually passed at call time.
    """
    static_params = [
        name
        for name, param in sig.parameters.items()
        if param.kind != Parameter.VAR_KEYWORD   # drop **kwargs placeholder
    ]
    return ["timestamp", "function"] + static_params + extra_kwarg_keys + ["result"]


def log_to_csv(csv_path: str):
    def decorator(func):
        sig = signature(func)

        # Identify whether this function accepts **kwargs and what its name is
        var_kw_name = next(
            (name for name, p in sig.parameters.items()
             if p.kind == Parameter.VAR_KEYWORD),
            None,
        )

        # Names of all *explicit* parameters (everything that isn't **kwargs)
        explicit_param_names = {
            name for name, p in sig.parameters.items()
            if p.kind != Parameter.VAR_KEYWORD
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # ------------------------------------------------------------------
            # 1. Build the dynamic schema for this specific call
            #    If the function has **kwargs, expand their keys into columns.
            # ------------------------------------------------------------------
            # Exclude kwarg keys that shadow explicit params
            extra_kwarg_keys = (
                sorted(k for k in kwargs.keys() if k not in explicit_param_names)
                if var_kw_name else []
            )
            expected_fieldnames = _build_schema(sig, extra_kwarg_keys)

            # ------------------------------------------------------------------
            # 2. Schema validation & filename search
            # ------------------------------------------------------------------
            base, ext = os.path.splitext(csv_path)
            current_path = csv_path
            counter = 1
            file_changed = False

            while True:
                if not os.path.exists(current_path):
                    break  # New file – schema is whatever we write first

                existing_header = _read_header(current_path)

                if existing_header is None:
                    # Empty / corrupt file – safe to (re)use
                    break

                if existing_header == expected_fieldnames:
                    break  # Schema matches

                # Mismatch: try next numbered filename
                file_changed = True
                current_path = f"{base}_{counter}{ext}"
                counter += 1

            # ------------------------------------------------------------------
            # 3. Warn if the filename was diverted
            # ------------------------------------------------------------------
            if file_changed:
                warnings.warn(
                    f"CSV schema mismatch for function '{func.__name__}'. "
                    f"Logging diverted to: {current_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # ------------------------------------------------------------------
            # 4. Prepare row data
            #    Bind all arguments, then flatten **kwargs into individual keys.
            # ------------------------------------------------------------------
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            row_data: dict[str, str] = {
                "timestamp": datetime.now().isoformat(),
                "function": func.__name__,
                "result": repr(result),
            }

            for name, val in bound.arguments.items():
                param = sig.parameters[name]
                if param.kind == Parameter.VAR_KEYWORD:
                    # Expand each kwarg key into its own column
                    for k, v in val.items():
                        row_data[k] = repr(v)
                else:
                    row_data[name] = repr(val)

            # ------------------------------------------------------------------
            # 5. Write to CSV
            # ------------------------------------------------------------------
            file_exists = os.path.isfile(current_path)
            with open(current_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=expected_fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)

            return result

        return wrapper
    return decorator