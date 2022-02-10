import logging

LEVEL_CORRESPONDANCE = {'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20, 'DEBUG': 10}

logger = None

def setup_custom_logger(level):
    """
    Init the logger for the application

    :param name: Name of the logger
    :param level: Level of the expected log
    :return: Logger
    """
    global logger

    if logger is None:        
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        logger = logging.getLogger("Cyanure")
        logger.setLevel(LEVEL_CORRESPONDANCE[level])
        logger.addHandler(handler)
    return logger
