import os
import lmfit


def save(obj, fname, overwrite='ask', **kwargs):
    """ Saves a `dump`-able object to a file.

    This is designed to reduce boilerplate at the command line when saving objects like
    `ModelResult`, `Model`, and `Parameters`
    """
    assert hasattr(obj, 'dump') and callable(obj.dump), "Object must have a dump method"
    mode = 'x'
    if os.path.exists(fname):
        if overwrite == 'ask' and input("Overwrite? (y/n)") == 'y':
            mode = 'w'
        elif overwrite is True:
            mode = 'w'
    else:
        mode = 'w'

    with open(fname, mode=mode) as f:
        obj.dump(f, **kwargs)


def load(fname, kind, **kwargs):
    """ Loads an lmfit object from disk

    This is designed to reduce boilerplate at the command line.
    Args:
         fname: the file name to open
         kind: the kind of object that is being opened: 'model', 'modelresult', or 'parameters'
         **kwargs: passed to the underlying load functions
    """
    kind = str(kind).lower().strip()
    if kind == 'model':
        return lmfit.model.load_model(fname, **kwargs)
    elif kind == 'modelresult':
        return lmfit.model.load_modelresult(fname, **kwargs)
    elif kind == 'parameters':
        with open(fname, 'r') as f:
            return lmfit.Parameters().load(f, **kwargs)
    else:
        raise ValueError(f"Unrecognized kind {kind}; try model, modelresult, or parameters")
