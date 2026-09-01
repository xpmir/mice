import logging


def setup_logging(level=logging.INFO, format=None):
    """
    Sets up centralized logging for the project and filters out noisy library logs.
    """
    if format:
        logging.basicConfig(level=level, format=format)
    else:
        logging.basicConfig(level=level)

    # Filter out noisy library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
