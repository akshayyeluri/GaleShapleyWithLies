import numpy as np
from gale_shapley import *

# formulas
# p_util = base + k * height
# p_util_exp = base + k * (height) + k * (eps - eps_mean)
# Can just define e[i] as eps[i] - eps.mean()

def run_single_par(eps, height_coeffs, p_utils, a_utils, verbose=False):
    """
    Run a modified gale shapley to generate matchings. We believe that
    proposer i's utility for a particular acceptor j can be described by the
    linear relation:
        u[i, j] = b[i, j] + k[i] * height[j]
    (aka utility of j for i is a base term b[i,j] + a proposer specific
    constant k[i] times an acceptor specific characteristic height[j].
    
    Now, the acceptors (being trash men) will lie about their height characteristic
    to try and trick the women. The women know the men are lying, and so subtract
    the expected value (mean) of these fibs from the reported height estimate the men
    give to get their expected utility for each man. In other words,
        expected_u[i, j] = b[i, j] + k[i] * (reported_height[j] - eps_mean)
    where reported_height[j] is the height an acceptor is reporting, and eps_mean
    is the mean of the fibs.

    Finally, noting reported_height[j] = height[j] + eps[j], we see that
        expected_u[i,j] = b[i,j] + k[i] * (height[j]) + k[i] * (eps[j] - eps_mean)
    Note that neither in u[i,j] nor expected_u[i,j] does it actually matter
    what eps_mean or eps[j] is, all that matters is their difference.
    
    Thus, if we define p_utils as the matrix of true utilities, 
    and eps[j] as the difference between man j's fib and the mean fib, then we
    have that expected_utils[i,j] = p_utils[i,j] + k[i] * eps[j]

    We then say Gale shapley works as the women chose the best man according to
    EXPECTED utility that hasn't rejected them yet. Before proposing, the women
    observe the actual utility, and only propose if the actual utility is higher
    than the expected utility of anyone else. Otherwise, the woman just corrects her
    expected utility of this man, and waits for the next round (call it a bad date)

    :param eps:
        vector of "fibs" for the acceptors, shape (n,). Actually, this is the de-meaned
        vector of fibs, can be thought of as the "number of inches past mean"
        a man lies about. 

    :param height_coeffs:
        vector of slope terms k[i] for each of the women, shape (n,)

    :param p_utils:
        matrix of proposer expected utils, shape (n,n). p_utils[i][j] is the expected utility
        of acceptor j for proposer i
        
    :param a_utils:
        matrix of acceptor utils, shape (n,n). a_utils[i][j] is the utility
        of proposer j for acceptor i

    :returns matching:
        array representing the matching. matching[i] is the acceptor proposer
        i gets paired with
    """
    n, _ = p_utils.shape

    assert (p_utils.shape == (n,n) and a_utils.shape == (n,n) 
            and height_coeffs.shape == (n,) and eps.shape == (n,))
    assert ((p_utils >= 0).all() and (a_utils >= 0).all() and np.isclose(eps.mean(), 0))

    # proposals[i][j] is True if acceptor j has a proposal from proposer i
    proposals = np.zeros((n,n), dtype=bool)

    # matching[i] is the acceptor proposer i is temporarily paired with
    matching = np.full(n, -1, dtype=int)

    # Compute the actual utils for proposers
    expected_p_utils = p_utils
    actual_p_utils   = p_utils - np.outer(height_coeffs, eps)


    # Maybe setup a progress bar
    if verbose:
        pbar = tqdm(total=n)

    while True:
        # Find all unmatched proposers
        curr_proposers = np.where(matching == -1)[0]

        # Stop if everyone is matched
        if len(curr_proposers) == 0:
            return matching

        # Find the best person for each curr_proposer that hasn't already gotten a proposal
        next_props = np.argmax(np.where(proposals, -1, expected_p_utils), axis=1)[curr_proposers]

        # Do the observe step (update the expected utils with the actual utils for
        # the guys about to be proposed to)
        expected_p_utils[curr_proposers, next_props] = actual_p_utils[curr_proposers, next_props]

        # Recompute who wants to propose (only those who still think the current guy is the best)
        next_props2 = np.argmax(np.where(proposals, -1, expected_p_utils), axis=1)[curr_proposers]
        idx = (next_props == next_props2)
        curr_proposers, next_props = curr_proposers[idx], next_props[idx]

        # Send out the proposals
        proposals[curr_proposers, next_props] = True

        # Find the acceptors that have at least 1 proposal
        matched_acceptors = np.where((proposals.T).any(axis=1))[0]

        # Find the best among the ppl that have proposed for each acceptor
        matched_proposers = np.argmax(np.where(proposals.T, a_utils, -1), axis=1)[matched_acceptors]

        # Update the matching
        new_matching = np.full(n, -1, dtype=int)
        new_matching[matched_proposers] = matched_acceptors
        this_round_matches = (new_matching != -1).sum() - (matching != -1).sum()
        matching = new_matching

        # Maybe update the progress bar with this_round_matches
        if verbose:
            pbar.update(this_round_matches)

