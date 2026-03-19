import time


PREFIX = "[UEDRS]"
TIME_START = time.time()


def log(*values: object, timestamp = True, start: float | None = None, sep = ' ', end = '\n'):
    if timestamp:
        current = time.time()
        elapsed = current - (start if start is not None else TIME_START)
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        timestamp_str = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
        print(PREFIX, timestamp_str, *values, sep=sep, end=end)
    else:
        print(PREFIX, *values, sep=sep, end=end)
