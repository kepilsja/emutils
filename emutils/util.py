import warnings
import functools
import csv
import os
from datetime import datetime
from inspect import signature

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.get(*args)
        return dotdict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def log_to_csv(csv_path: str):
    def decorator(func):
        sig = signature(func)
        # 1. Define the expected schema
        expected_fieldnames = ["timestamp", "function"] + list(sig.parameters.keys()) + ["result"]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 2. Schema Validation and Filename Search
            base, ext = os.path.splitext(csv_path)
            current_path = csv_path
            counter = 1
            file_changed = False
            
            while True:
                if not os.path.exists(current_path):
                    break
                
                try:
                    with open(current_path, mode="r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        existing_header = next(reader)
                        
                        if existing_header == expected_fieldnames:
                            break  # Schema matches!    
                        else:
                            # Mismatch detected
                            file_changed = True
                            current_path = f"{base}_{counter}{ext}"
                            counter += 1
                except (StopIteration, IOError):
                    # File exists but is empty/corrupt; we can use it
                    break

            # 3. Issue Warning if the filename was diverted
            if file_changed:
                warnings.warn(
                    f"CSV Schema mismatch for function '{func.__name__}'. "
                    f"Logging diverted to: {current_path}",
                    RuntimeWarning
                )

            # 4. Prepare and Write Data
            bound_arguments = sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            
            row_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "function": func.__name__,
                "result": repr(result),
                **{name: repr(val) for name, val in bound_arguments.arguments.items()}
            }

            file_exists = os.path.isfile(current_path)
            with open(current_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=expected_fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)

            return result
        return wrapper
    return decorator