"""Session-only experiment helper. Not a production cache or workflow engine."""
_entries = {}


def once(key, create, *, healthy=None, close=None):
    if key not in _entries:
        value = create()  # Only successful returns enter the cache.
        _entries[key] = (value, close)
    value, _ = _entries[key]
    if healthy is not None and not healthy(value):
        raise RuntimeError(f'{key}: retained resource is unhealthy; invalidate explicitly')
    return value


def invalidate(key):
    if key in _entries:
        value, close = _entries[key]
        if close is not None:
            close(value)  # Keep the entry if cleanup fails, so it remains reachable.
        del _entries[key]
