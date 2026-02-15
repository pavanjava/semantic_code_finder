import time


def get_extension(language):
    """Get file extension for a programming language."""
    extensions = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java"
    }
    return extensions.get(language, f".pf")


# decorator function to compute the execution time
def compute_time_elapsed():
    def wrapper(func):
        def inner(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            print(f"Function '{func.__name__}' took {elapsed:.4f} seconds")
            return result  # ← Missing in your code
        return inner  # ← Missing in your code
    return wrapper