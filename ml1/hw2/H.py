from xml.sax.xmlreader import InputSource
import numpy as np


def get_best_indices(ranks: np.ndarray, top: int, axis: int = 1) -> np.ndarray:
    """
    Returns indices of top largest values in rows of array.

    Parameters
    ----------
    ranks : np.ndarray, required
        Input array.
    top : int, required
        Number of largest values.
    """
    ind = np.argpartition(ranks, -top, axis=axis)
    ind = np.take(np.flip(ind, axis=axis), np.arange(top), axis=axis)
    input = np.take_along_axis(ranks, ind, axis=axis)
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    return np.flip(ind, axis=axis)


if __name__ == "__main__":
    with open('input.bin', 'rb') as f_data:
        ranks = np.load(f_data)
    indices = get_best_indices(ranks, 5)
    with open('output.bin', 'wb') as f_data:
        np.save(f_data, indices)
