import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from soynlp.tokenizer import MaxScoreTokenizer


def diverse_keysentences(keywords, texts, topk=10, diversity=0.3, penalty=None, raw_texts=None):
    """
    Arguments
    ---------
    keywords : {str:int}
        {word:rank} trained from KR-WordRank.
        texts will be tokenized using keywords
    texts : list of str
        Each str is a sentence.
    topk : int
        Number of key sentences
    diversity : float
        Minimum cosine distance between top ranked sentence and others.
        Large value makes this function select various sentence.
        The value must be [0, 1]
    penalty : callable
        Penalty function. str -> float
        Default is no penalty
        If you use only sentence whose length is in [25, 40],
        set penalty like following example.
            >>> penalty = lambda x: 0 if 25 <= len(x) <= 40 else 1
    raw_texts : list of str or None

    Returns
    -------
    keysentences : list of str
    """

    if isinstance(raw_texts, list):
        assert len(texts) == len(raw_texts)

    if not callable(penalty):
        penalty = lambda x: 0

    if not 0 <= diversity <= 1:
        raise ValueError('Diversity must be [0, 1] float value')

    vectorizer = KeywordVectorizer(keywords)
    x = vectorizer.vectorize(texts)
    if x.nnz == 0:
        return []

    keyvec = vectorizer.keyword_vector.reshape(1,-1)
    initial_penalty = np.asarray([penalty(sent) for sent in texts])
    idxs = select_keysentences(x, keyvec, texts, initial_penalty, topk, diversity)

    if raw_texts is None:
        raw_texts = texts

    keysentences = [raw_texts[idx] for idx in idxs]
    return keysentences

def select_keysentences(x, keyvec, texts, initial_penalty, topk=10, diversity=0.3):
    """
    Arguments
    ---------
    x : scipy.sparse.csr_matrix
        (n docs, n keywords) Boolean matrix
    keyvec : numpy.ndarray
        (1, n keywords) rank vector
    texts : list of str
        Each str is a sentence
    initial_penalty : numpy.ndarray
        (n docs,) shape. Defined from penalty function
    topk : int
        Number of key sentences
    diversity : float
        Minimum cosine distance between top ranked sentence and others.
        Large value makes this function select various sentence.
        The value must be [0, 1]

    Returns
    -------
    keysentence indices : list of int
        The length of keysentences is topk at most.
    """

    diversity = diversity + 0.00001 # for truncation error

    dist = pairwise_distances(x, keyvec, metric='cosine').reshape(-1)
    dist = dist + initial_penalty

    idxs = []
    for _ in range(topk):
        idx = dist.argmin()
        idxs.append(idx)
        dist[idx] += 2 # maximum distance of cosine is 2
        idx_all_distance = pairwise_distances(
            x, x[idx].reshape(1,-1), metric='cosine').reshape(-1)
        penalty = np.zeros(idx_all_distance.shape[0])
        penalty[np.where(idx_all_distance <= diversity)[0]] = 2
        dist += penalty
    return idxs


class KeywordVectorizer:
    """
    Arguments
    ---------
    vocab_score : dict
        {str:float} form keyword vector

    Attributes
    ----------
    tokenizer : soynlp.tokenizer.MaxScoreTokenizer
        Tokenizer instance
    idx_to_vocab : list of str
        Vocab list
    vocab_to_idx : dict
        {str:int} Vocab to index mapper
    keyword_vector : numpy.ndarray
        shape (len(idx_to_vocab),) vector
    """

    def __init__(self, vocab_score):
        self.tokenizer = MaxScoreTokenizer(scores=vocab_score)
        self.idx_to_vocab = [vocab for vocab in sorted(vocab_score, key=lambda x:-vocab_score[x])]
        self.vocab_to_idx = {vocab:idx for idx, vocab in enumerate(self.idx_to_vocab)}
        self.keyword_vector = np.asarray(
            [score for _, score in sorted(vocab_score.items(), key=lambda x:-x[1])])

        def l2_normalize(vectors):
            return vectors / np.sqrt((vectors ** 2).sum())

        self.keyword_vector = l2_normalize(self.keyword_vector)

    def tokenize(self, sent):
        terms = self.tokenizer.tokenize(sent)
        terms = [term for term in terms if term in vocab_to_idx]
        return terms

    def vectorize(self, sents):
        """
        Argument
        --------
        sents : list of str
            Each str is sentence

        Returns
        -------
        scipy.sparse.csr_matrix
            (n sents, n keywords) shape Boolean matrix
        """
        rows, cols, data = [], [], []
        for i, sent in enumerate(sents):
            terms = set(self.tokenize(sent))
            for term in terms:
                j = self.vocab_to_idx.get(term, -1)
                if j == -1:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(1)
        n_docs = len(sents)
        n_terms = len(self.idx_to_vocab)
        return csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms))
