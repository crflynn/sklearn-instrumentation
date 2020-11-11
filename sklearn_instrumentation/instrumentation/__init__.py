from functools import wraps


def identity(func, **dkwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
