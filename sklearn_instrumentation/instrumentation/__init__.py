from sklearn_instrumentation.utils import wraps


def identity(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
