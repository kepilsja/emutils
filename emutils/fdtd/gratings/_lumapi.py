"""Lumerical API classes, replaced by `object` if lumapi is unavailable."""
import logging

from ...lumer.lumapi_loader import add_lumapi_to_path

logger = logging.getLogger(__name__)

try:
    add_lumapi_to_path()
    import lumapi
    from lumapi import MODE as LumerMODE
    from lumapi import FDTD as LumerFDTD

except ImportError as err:
    logger.warning("Lumerical unavailable!")
    logger.warning(err)

    LumerMODE = object
    LumerFDTD = object
