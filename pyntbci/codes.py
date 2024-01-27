import numpy as np


def is_de_bruijn_sequence(code, k=2, n=6):
    """Check whether a code is a de Bruijn sequence. A de Bruijn sequence [1]_ should contain all possible substrings
    of the alphabet.

    Parameters
    ----------
    code: np.ndarray
        A vector with the de Bruijn sequence of shape (1, n_bits).
    k: int (default: 2)
        The size of the alphabet.
    n: int (default: 6)
        The order of the sequence.

    Returns
    -------
    out: bool
        True if the code is a de Bruijn sequence, otherwise False.

    References
    ----------
    .. [1] De Bruijn, N. G. (1946). A combinatorial problem. Proceedings of the Section of Sciences of the Koninklijke
           Nederlandse Akademie van Wetenschappen te Amsterdam, 49(7), 758-764.
    """
    code = code.flatten()
    n_bits = code.size

    # Check the length of the code
    if n_bits != k**n:
        return False

    # Initialize bitset
    seen = set()
    seen.add(k**n)

    # Initialize current to correspond to the word formed by the (n-1) 
    # last elements
    current = 0
    for i in range(n-1):
        current = k * current + code[-n + i + 1]

    # Stop if the same word has been met twice
    for i in code:
        current = (k * current + i) % n_bits
        if current in seen or i < 0 or i >= k:
            return False
        seen.add(current)
    return True


def is_gold_codes(codes):
    """Check whether the codes are Gold codes. Gold codes [3]_ have a 3-valued auto- and cross-correlation function
    [4]_. If the length of the linear feedback shift register m is even:
    * 1/(2^n−1) 
    * −(2^{(n+2)/2}+1)/(2^n−1)
    * (2^{(n+2)/2}−1)/(2^n−1) 
    in ratio ~3/4, ~1/8, ~1/8.
    If the length of the linear feedback shift register n is odd:
    * 1/(2^n−1)
    * −(2^{(n+1)/2}+1)/(2^n−1)
    * (2^{(n+1)/2}−1)/(2^n−1) 
    in ratio ~1/2, ~1/4, ~1/4

    Parameters
    ----------
    codes: np.ndarray
        A vector with the m-sequence of shape (n_codes, n_bits).

    Returns
    -------
    out: bool
        True if the codes are an Gold codes, otherwise False.

    References
    ----------
    .. [3] Gold, R. (1967). Optimal binary sequences for spread spectrum multiplexing (Corresp.). IEEE Transactions on
           information theory, 13(4), 619-621.
    .. [4] Meel, J. (1999). Spread spectrum (SS). De Nayer Instituut, Hogeschool Voor Wetenschap & Kunst.
    """
    assert np.unique(codes).size == 2, "The input sequences are not binary."
    n_codes, n_bits = codes.shape
    n = int(np.log2(n_bits + 1))

    # Binary to bipolar
    codes = codes.astype("int8")
    codes = 2 * codes - 1

    # Compute correlations
    rho = np.zeros((n_codes, n_codes, n_bits))
    for i in range(n_codes):
        for j in range(n_codes):
            for k in range(n_bits):
                shifted = np.roll(codes[i, :], k)
                rho[i, j, k] = np.sum(codes[j, :] * shifted) / n_bits

    # Check correlations
    unique = np.unique(np.round(rho, 6))
    cond1 = len(unique) == 4
    if not cond1:
        return False
    if n % 2 == 0:
        cond2 = unique[0] == np.round(-(2**((n+2)/2)+1)/n_bits, 6)
        cond3 = unique[1] == np.round(-1/n_bits, 6)
        cond4 = unique[2] == np.round((2**((n+2)/2)-1)/n_bits, 6)
        cond5 = unique[3] == 1.
    else:
        cond2 = unique[0] == np.round(-(2**((n+1)/2)+1)/n_bits, 6)
        cond3 = unique[1] == np.round(-1/n_bits, 6)
        cond4 = unique[2] == np.round((2**((n+1)/2)-1)/n_bits, 6)
        cond5 = unique[3] == 1.
    return cond1 and cond2 and cond3 and cond4 and cond5


def is_m_sequence(code):
    """Check whether a code is an m-sequence. An m-sequence [5]_ should have an auto-correlation function that is 1 at
    time-shift 0 and -1/n elsewhere [6]_.

    Parameters
    ----------
    code: np.ndarray
        A vector with the m-sequence of shape (1, n_bits).

    Returns
    -------
    out: bool
        True if the code is an m-sequence, otherwise False.

    References
    ----------
    .. [5] Golomb, S. W. (1967). Shift register sequences. Holden-Day. Inc., San Fransisco.
    .. [6] Meel, J. (1999). Spread spectrum (SS)
    """
    code = code.flatten()
    n_bits = code.size

    # Binary to bipolar
    code = code.astype("int8")
    code = 2*code - 1

    # Compute correlations
    rho = np.zeros(n_bits)
    for i in range(n_bits):
        rho[i] = np.sum(code * np.roll(code, i)) / n_bits

    # Check correlations
    unique = np.unique(np.round(rho, 6))
    cond1 = unique.size == 2  # two-valued
    cond2 = unique[0] == np.round(-1/n_bits, 6)  # other shifts are -1/n
    cond3 = unique[1] == 1.  # zero-shift is 1
    return cond1 and cond2 and cond3
    

def make_apa_sequence():
    """Generate an almost perfect auto-correlation sequence. APA sequence [7]_ examples are taken from [8]_.

    Returns
    -------
    code: (numpy.ndarray):
        A matrix with an APA sequence of shape (1, n_bits).

    References
    ----------
    .. [7] Wolfmann, J. (1992). Almost perfect autocorrelation sequences. IEEE Transactions on Information Theory,
           38(4), 1412-1418. DOI: 10.1109/18.144729
    .. [8] Wei, Q., Liu, Y., Gao, X., Wang, Y., Yang, C., Lu, Z., & Gong, H. (2018). A novel c-VEP BCI paradigm for
           increasing the number of stimulus targets based on grouping modulation with different codes. IEEE
           Transactions on Neural Systems and Rehabilitation Engineering, 26(6), 1178-1187.
           DOI: 10.1109/TNSRE.2018.2837501
    """
    # Credit: Wei et al. (2018) doi: 10.1109/TNSRE.2018.2837501
    code = [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    code = np.array(code).astype("uint8")[np.newaxis, :]
    return code


def make_de_bruijn_sequence(k=2, n=6, seed=None):
    """Generate a de Bruijn sequence. The code to generate de Bruijn sequences [9]_ is largely inspired by [10]_.

    Parameters
    ----------
    k: int (default: 2)
        The size of the alphabet.
    n: int (default: 6)
        The order of the sequence.
    seed: list (default: None)
        Seed for the initial register. None leads to an all zero initial register.

    Returns
    -------
    code: np.ndarray
        A matrix with a de Bruijn sequence of shape (1, n_bits).

    References
    ----------
    .. [9] De Bruijn, N. G. (1946). A combinatorial problem. Proceedings of the Section of Sciences of the Koninklijke
           Nederlandse Akademie van Wetenschappen te Amsterdam, 49(7), 758-764.
    .. [10] Eviatar Bach: git.sagemath.org/sage.git/tree/src/sage/combinat/debruijn_sequence.pyx
    """
    if seed is None:
        register = [0] * k * n   
    else:
        register = seed
    assert len(register) == n * k, "The register must be of length n*k."  
    alphabet = list(range(k))

    def db(seq, reg, t, p):
        if t > n:
            if n % p == 0:
                seq.extend(reg[1: p + 1])
        else:
            reg[t] = reg[t - p]
            seq = db(seq, reg, t + 1, p)
            for j in range(reg[t - p] + 1, k):
                reg[t] = j
                seq = db(seq, reg, t + 1, t)
        return seq

    sequence = db([], register, 1, 1)
    code = np.array([alphabet[i] for i in sequence])[np.newaxis, :]
    return code


def make_golay_sequence():
    """Generate complementary Golay sequences. Golay sequence [11]_ examples are taken from [12]_.

    Returns
    -------
    codes: np.ndarray
        A matrix with two complementary Golay sequences of shape (2, n_bits).

    References
    ----------
    .. [11] Golay, MJE. (1949). Notes on digital coding. Proc. IEEE, 37, 657.
    .. [12] Wei, Q., Liu, Y., Gao, X., Wang, Y., Yang, C., Lu, Z., & Gong, H. (2018). A novel c-VEP BCI paradigm for
            increasing the number of stimulus targets based on grouping modulation with different codes. IEEE
            Transactions on Neural Systems and Rehabilitation Engineering, 26(6), 1178-1187. DOI:
            10.1109/TNSRE.2018.2837501
    """
    # Credit: Wei et al. (2018) doi: 10.1109/TNSRE.2018.2837501
    ga = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1,
          1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    gb = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
          0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    codes = np.array([ga, gb]).astype("uint8")
    return codes


def make_gold_codes(poly1=(1, 0, 0, 0, 0, 1), poly2=(1, 1, 0, 0, 1, 1), seed1=None, seed2=None):
    """Generate a set of Gold codes. The Gold codes [13]_ should be generate with two polynomials that define a
    preferred pair of m-sequences.
    
    Parameters
    ----------
    poly1: tuple (default: (1, 0, 0, 0, 0, 1))
        The feedback tap points defined by the primitive polynomial.
        Example: 1 + x + x^6 is represented as (1, 0, 0, 0, 0, 1) and 1 + 4x + 3x^2 as (4, 3).
    poly2: tuple (default: (1, 1, 0, 0, 1, 1))
        The feedback tap points defined by the primitive polynomial.
        Example: 1 + x + x^6 is represented as (1, 0, 0, 0, 0, 1) and 1 + 4x + 3x^2 as (4, 3).
    seed1: list (efault: None)
        Seed for the initial register for poly1. None leads to an all zero initial register.
    seed2: list (default: None)
        Seed for the initial register for poly1. None leads to an all zero initial register.
        
    Returns
    -------
    codes: np.ndarray
        A matrix with Gold codes of shape (n_codes, n_bits).

    References
    ----------
    .. [13] Gold, R. (1967). Optimal binary sequences for spread spectrum multiplexing (Corresp.). IEEE Transactions on
            information theory, 13(4), 619-621.
    """
    assert np.unique(np.array(poly1)).size == 2, "The poly1 is not binary."
    assert np.unique(np.array(poly2)).size == 2, "The poly2 is not binary."
    n = len(poly1)
    assert n == len(poly2), "Both polynomials should be the same length."
    m_sequence1 = make_m_sequence(poly1, 2, seed1).flatten()
    m_sequence2 = make_m_sequence(poly2, 2, seed2).flatten()
    codes = np.empty((2**n-1, 2**n-1), dtype="uint8")
    for i in range(2**n-1):
        codes[i, :] = (m_sequence1 + m_sequence2) % 2
        m_sequence2 = np.roll(m_sequence2, -1)
    return codes


def make_m_sequence(poly=(1, 0, 0, 0, 0, 1), base=2, seed=None):
    """Generate a maximum length sequence. Maximum length sequence, or m-sequence [14]_.
    
    Parameters
    ----------
    poly: tuple (default: (1, 0, 0, 0, 0, 1))
        The feedback tap points defined by the primitive polynomial.
        Example: 1 + x + x^6 is represented as (1, 0, 0, 0, 0, 1) and 1 + 4x + 3x^2 as (4, 3).
    base: int (default: 2)
        The base of the sequence (related to the Galois Field), i.e. base 2 generates a binary sequence, base 3 a
        tertiary sequence, etc.
    seed: list (default: None)
        The seed for the initial register. None leads to an all zero initial register.
        
    Returns
    -------
    code: np.ndarray
        A matrix with an m-sequence of shape (1, n_bits).

    References
    ----------
    .. [14] Golomb, S. W. (1967). Shift register sequences. Holden-Day. Inc., San Fransisco.
    """
    n = len(poly)
    poly = np.array(poly)
    assert np.all(poly < base), "All values in the polynomial should be smaller than the base."
    if seed is None:
        register = np.ones(n, dtype="uint8")
    else:
        register = np.array(seed, dtype="uint8")
    assert len(register) == n, "The (seeded) register must be of length n (of the polynomial)."
    code = np.zeros(2**n-1, dtype="uint8")
    for i in range(2**n-1):
        bit = np.sum(poly * register) % base
        register = np.roll(register, 1)
        register[0] = bit
        code[i] = bit
    return code[np.newaxis, :]


def modulate(codes):
    """Modulate a set of codes. Modulation is done by xoring with a double frequency bit-clock [15]_. This limits
    low-frequency content as well as the event distribution (i.e., limits to shorter (only two) run-lengths).
    
    Parameters
    ----------
    codes: np.ndarray
        A matrix with codes of shape (n_codes, n_bits).

    Returns
    -------
    codes: np.ndarray
        A matrix with modulated codes of shape (n_codes, 2 * n_bits).

    References
    ----------
    .. [15] Thielen, J., van den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
            re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797. DOI: 10.1371/journal.pone.0133797
    """
    codes = np.repeat(codes, 2, axis=1)
    clock = np.zeros(codes.shape, dtype="uint8")
    clock[:, ::2] = 1
    return (codes + clock) % 2
    