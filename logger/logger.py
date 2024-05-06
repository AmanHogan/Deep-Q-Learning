""" Logs data to logfile.txt """
import time

def log(message, to_console_only=None) -> None:
    """
    Logs messages to log.txt on robot and prints to console as well\n
    Args:
        message (str): msg to log
        to_console_only (Bool, optional): True if you want
        to print message only to the console. Defaults to None.
    """

    current_time = time.time()
    local_time = time.localtime(current_time)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    log_message = "{}: {}".format(timestamp, message)  # Format the log message

    with open('./logger/log.txt', 'a') as log_file:
        log_file.write(log_message + '\n')

# Example usage:
if __name__ == "__main__":
    log("This is a log message.")