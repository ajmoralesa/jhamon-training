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
    import pickle

    with open((path / name), "rb") as f:
        return pickle.load(f)
