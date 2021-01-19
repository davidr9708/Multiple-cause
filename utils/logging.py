from datetime import datetime
import pandas as pd


def ymd_timestamp(time=datetime.now()):
    """Timestamp for saving phase outputs."""
    return '{:%Y_%m_%d}'.format(time)


def print_memory_timestamp(df, message, time=datetime.now()):
    """Print memory usage and timestamp."""
    print(f"[{datetime.now()}] {message} \nmemory in bytes: \n{df.memory_usage(deep=True)}")
