from datetime import datetime


def timer(func: callable) -> callable:
    """
    Decorator used for time measuring.
    -----
    :param func: callable
        function, that need to be wrapped
    -----
    :return: callable
        returns wrapper function which prints time used for executing func and returns some result
    """
    def wrapper(*args: object, **kwargs: object) -> object:
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        time = end - start
        print(f"{func.__name__}() executed in {time}.")
        return result
    return wrapper
