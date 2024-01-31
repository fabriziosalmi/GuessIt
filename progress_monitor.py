import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_progress_monitor(total_steps):
    """
    Set up a tqdm progress bar.
    :param total_steps: Total number of steps in the process.
    :return: tqdm progress bar object.
    """
    return tqdm(total=total_steps, desc="Progress", ncols=100, bar_format='{l_bar}{bar}{r_bar}')

def log_info(message):
    """
    Log an informational message.
    :param message: Message to log.
    """
    logging.info(message)

def log_warning(message):
    """
    Log a warning message.
    :param message: Message to log.
    """
    logging.warning(message)

def log_error(message):
    """
    Log an error message.
    :param message: Message to log.
    """
    logging.error(message)

def log_debug(message):
    """
    Log a debug message.
    :param message: Message to log.
    """
    logging.debug(message)

def update_progress(progress_bar, step=1):
    """
    Update the progress bar.
    :param progress_bar: tqdm progress bar object.
    :param step: Number of steps to update.
    """
    progress_bar.update(step)

def close_progress(progress_bar):
    """
    Close the progress bar.
    :param progress_bar: tqdm progress bar object.
    """
    progress_bar.close()
