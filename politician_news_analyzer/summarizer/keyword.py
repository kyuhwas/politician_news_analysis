import numpy as np


def proportion_keyword(bow, pos_idx, idx_to_vocab, ref_idx=None, topk1=100, topk2=30):
    """
    Arguments
    ---------:
    bow : scipy.sparse.csr_matrix
        (n_docs, n_terms) shape Bag-of-Words Model
    pos_idx : list, set, int, or numpy.ndarray
        Positive row index
    idx_to_vocab : list of str
        Vocabulary list
    ref_idx : list, set, int, numpy.ndarray, or None
        If None, use rest row of pos_idx
    topk1: int
        Number of most frequent terms for keyword candidates
    topk2: int
        Number of selected keywords

    Returns
    -------
    keywords : list of tuple
        Each tuple consists of (keyword, score, proportion in positive documents)
    """

    n_docs, n_terms = bow.shape

    # type check
    pos_idx = type_check_index(pos_idx)

    # default ref_idx
    if ref_idx is None:
        ref_idx = [i for i in range(n_docs) if not (i in pos_idx)]
    ref_idx = type_check_index(ref_idx)

    pos = to_proportion(bow, pos_idx)
    ref = to_proportion(bow, ref_idx)
    ratio = proportion_ratio(pos, ref)

    # select candidates (frequent terms)
    candidates_idx = pos.argsort()[-topk1:]
    candidates_score = ratio[candidates_idx]

    # sort by distinctness
    keyword_idx = candidates_idx[candidates_score.argsort()[-topk2:]]
    keywords = [(idx, ratio[idx]) for idx in keyword_idx]
    keywords = [(idx_to_vocab[idx], score, pos[idx]) for idx, score in reversed(keywords)]

    return keywords

def type_check_index(idx):
    if isinstance(idx, list):
        idx = np.asarray(idx, dtype=np.int)
    elif isinstance(idx, set):
        idx = np.asarray(list(idx), dtype=np.int)
    elif isinstance(idx, int):
        idx = np.asarray([idx], dtype=np.int)
    elif not isinstance(idx, np.ndarray):
        raise ValueError('idx must be list, int or numpy.ndarray')
    return idx

def to_proportion(bow, doc_idx):
    prop = np.asarray(bow[doc_idx,:].sum(axis=0))[0]
    prop = prop / prop.sum()
    return prop

def proportion_ratio(pos, ref):
    assert len(pos) == len(ref)
    ratio = pos / (pos + ref)
    ratio = np.nan_to_num(ratio)
    return ratio
