import sys
import logging

from .lumer.lumapi_loader import add_lumapi_to_path

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
