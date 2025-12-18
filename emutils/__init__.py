import sys
import logging

from .lumer.lumapi_loader import add_lumapi_to_path
add_lumapi_to_path()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

um, nm = 1e-6, 1e-9