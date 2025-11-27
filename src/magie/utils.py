import inspect
import functools


def enforce_types(**type_map):
    """
    Lightweight runtime argument type checking.

    Usage:
        @enforce_types(path=str, site_code=str, opts=(dict, type(None)))
        def func(path, site_code, opts=None, **kwargs):
            ...

    Each key is the parameter name; each value is either a single type
    or a tuple of allowed types.
    """
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            for name, expected in type_map.items():
                if name not in bound.arguments:
                    continue

                value = bound.arguments[name]
                # Allow None when explicitly included via type(None)
                if isinstance(expected, tuple):
                    ok = isinstance(value, expected)
                    expected_names = ", ".join(t.__name__ for t in expected)
                else:
                    ok = isinstance(value, expected)
                    expected_names = expected.__name__

                if not ok:
                    raise TypeError(
                        f"Argument '{name}' to {func.__name__}() must be instance of "
                        f"{expected_names}, got {type(value).__name__}"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator
# Handling import errors for GitHub repositories
def validinput(inputstr, positive_answer, negative_answer):
    answer = input(inputstr+'\n')
    if answer == positive_answer:
        return True
    elif answer == negative_answer:
        return False
    else:
        print('Invalid response should be either' + str(positive_answer) + ' or ' + str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)