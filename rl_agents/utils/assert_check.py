from rl_agents.utils.wrapper import Wrapper

def assert_is_instance(value, expected, name: str | None = None, skip_wrappers = True):
    """Assert isinstance(value, expected) with a clear, auto-qualified error."""

    # Skip wrapper classes to retrieve initial module
    if skip_wrappers:
        while isinstance(value, Wrapper):
            value = value.wrapped
    
    # Case 1 : Value respect the expected class -> return the value
    if isinstance(value, expected):
        return value  # allow: x = assert_is_instance(x, T)

    # Case 2 : value does not respect the expected class -> raised error message
    exp  = _qual(expected)
    got  = _qual(type(value))
    is_proto = (getattr(expected, "_is_protocol", False) or
                (isinstance(expected, tuple) and any(getattr(x, "_is_protocol", False) for x in expected)))
    verb = "implement" if is_proto else "be instance of"
    raise AssertionError(f"{name or 'value'} must {verb} {exp} (got {got}).")

def _qual(t):
    if isinstance(t, tuple):
        return "(" + ", ".join(_qual(x) for x in t) + ")"
    mod = getattr(t, "__module__", None)
    qn  = getattr(t, "__qualname__", getattr(t, "__name__", repr(t)))
    return qn if not mod or mod == "builtins" else f"{mod}.{qn}"