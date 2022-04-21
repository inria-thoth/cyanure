"""Contain the logger configuration used in the project."""

import logging

LEVEL_CORRESPONDANCE = {'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20, 'DEBUG': 10}

LOGGER = None


def setup_custom_logger(level):
    """
    Init the logger for the application.

    :param level: Level of the expected log
    :return: Logger instance
    """
    global LOGGER

    if LOGGER is None:
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        LOGGER = logging.getLogger("Cyanure")
        LOGGER.setLevel(LEVEL_CORRESPONDANCE[level])
        LOGGER.addHandler(handler)
    return LOGGER
