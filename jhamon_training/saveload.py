# These functions assume that you have an obj folder in your current working
# directory, which will be used to store the objects.
# Note that pickle.HIGHEST_PROTOCOL is a binary format, which could not be always
# convenient, but is good for performance. Protocol 0 is a text format.

import os
import pickle


def save_obj(obj, path):
    # Ensure the directory exists before trying to save the file
    directory = os.path.dirname(path)
    if directory:  # Ensure directory is not an empty string if path is just a filename
        os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    """Load a pickled object, tolerating pickles written by older pandas.

    The cached ``nht_results.pkl`` / ``ikt_results.pkl`` were written under
    pandas <2.0 and contain DataFrames whose internal blocks pickle a raw
    ``slice`` where pandas >=2.0 expects a ``BlockPlacement``. A plain
    ``pickle.load`` therefore dies with::

        TypeError: Argument 'placement' has incorrect type
                   (expected pandas._libs.internals.BlockPlacement, got slice)

    ``pd.read_pickle`` routes through pandas' own compatibility unpickler,
    which rewrites those blocks on the way in, so it reads both the legacy and
    the current caches. Falling back to it (rather than replacing outright)
    keeps non-pandas payloads working exactly as before.
    """
    import pandas as pd

    target = path / name
    with open(target, "rb") as f:
        try:
            return pickle.load(f)
        except TypeError:
            pass

    return pd.read_pickle(target)
