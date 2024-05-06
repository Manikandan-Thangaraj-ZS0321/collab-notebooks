import logging
from datetime import datetime
import os


class Logger:
    def __init__(self, logger_name, log_level=logging.INFO, log_to_file=True, log_directory="./"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # Configure the logger to prevent log propagation
        self.logger.propagate = False

        formatter = logging.Formatter('%(asctime)s %(process)d [%(levelname)s] [%(threadName)s] %(thread)d %('
                                      'filename)s: %(lineno)s: %(message)s')

        if log_to_file and log_directory is None:
            log_directory = os.getcwd()  # Set log directory to the current working directory

            # Create the log directory if it doesn't exist
            os.makedirs(log_directory, exist_ok=True)

            # Include date and time in the log file name
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_name = f"{logger_name}_{current_datetime}.log"
            log_file_path = os.path.join(log_directory, log_file_name)

            try:
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(log_level)

                # Add a different formatter for file logs with timestamp
                file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(threadName)s %(filename)s:%('
                                                   'lineno)s - %(message)s')
                file_handler.setFormatter(file_formatter)

                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Error setting up file logger: {e}")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


# Initialize a custom logger for Triton
llama_logger = Logger('Llama3_logger', log_level=logging.INFO, log_to_file=True)
logger = llama_logger.get_logger()

if __name__ == "__main__":
    llama_logger = Logger("Llama3_logger", log_level=logging.INFO, log_to_file=True, log_directory="./")
    logger = llama_logger.get_logger()

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
