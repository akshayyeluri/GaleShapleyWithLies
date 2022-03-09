import numpy as np
import itertools
from tqdm import tqdm

__all__ = ['utils_from_rankings', 'rankings_from_utils', 'invert_matching', 'check_stability', 'run_gale_shapley']

def invert_matching(matching):
    """
    Invert a matching (return m_tilde, where m_tilde[i] = j if matching[j] = i)
    """
    n = len(matching)
    matching_inv = np.full(n, -1, dtype=int)
    matching_inv[matching] = np.arange(n, dtype=int)
    return matching_inv

def utils_from_rankings(rankings):
    """
    Take a rankings matrix where rankings[i] is a permutation of
    0...n-1 from favorite to least favorite, and convert it to a utility
    matrix representing the same rankings
    """
    n, _ = rankings.shape
    idx = np.arange(n, dtype=int)
    utils = np.zeros((n,n))
    # For person i, utility of 1st favorite person is n-1, 2nd fav is n-2, least fav is 0
    utils[idx[:, None], rankings[:, ::-1]] = idx[None, :]
    return utils

def rankings_from_utils(utils):
    """
    Convert a utility matrix where utils[i][j] is the utility of person j to person i
    to a rankings matrix where row i is a permutation of the people 0...n-1 from 
    favorite to least
    """
    return np.argsort(utils, axis=1)[:, ::-1]


def check_stability(matching, p_utils, a_utils):
    """
    Check that makes sure a matching is stable, i.e. no one wants to cheat.

    Formally, no proposer p1 and acceptor a1 exist such that
    p1's utility for their partner is less than their utility for a1,
    and a1's utility for their partner is less than their utility for p1
    """
    n = len(matching)

    assert (p_utils.shape == (n,n) and a_utils.shape == (n,n))
    assert ((p_utils >= 0).all() and (a_utils >= 0).all())
    assert (np.sort(matching) == np.arange(n, dtype=int)).all()

    matching_inv = invert_matching(matching)

    for p1 in range(n):
        for a1 in range(n):
            a2, p2 = matching[p1], matching_inv[a1]
            if (p_utils[p1, a2] < p_utils[p1, a1] and a_utils[a1, p2] < a_utils[a1, p1]):
                return False
    return True


def run_gale_shapley(p_utils, a_utils, verbose=False):
    """
    Run the gale shapley algorithm to generate a matching

    :param p_utils:
        matrix of proposer utils, shape (n,n). p_utils[i][j] is the utility
        of acceptor j for proposer i
        
    :param a_utils:
        matrix of acceptor utils, shape (n,n). a_utils[i][j] is the utility
        of proposer j for acceptor i

    :returns matching:
        array representing the matching. matching[i] is the acceptor proposer
        i gets paired with
    """
    n, _ = p_utils.shape

    assert (p_utils.shape == (n,n) and a_utils.shape == (n,n))
    assert ((p_utils >= 0).all() and (a_utils >= 0).all())

    # proposals[i][j] is True if acceptor j has a proposal from proposer i
    proposals = np.zeros((n,n), dtype=bool)

    # matching[i] is the acceptor proposer i is temporarily paired with
    matching = np.full(n, -1, dtype=int)

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
        next_props = np.argmax(np.where(proposals, -1, p_utils), axis=1)[curr_proposers]

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


def trial(n):
    """
    Run a trial with a number of ppl n, generating the matchings when running 
    gale shapley with men proposing, and with women proposing. Check both
    matchings are stable, and ensure that the utility of matches
    for men when men are proposing is higher than men's utilities in the matching
    where women are proposing. Show the same holds vice versa (women proposing
    is better for women)
    """
    idx = np.arange(n, dtype=int)
    men_utils = np.random.rand(n,n)
    women_utils = np.random.rand(n,n)
    
    men_prop_m2w   = run_gale_shapley(men_utils, women_utils)
    men_prop_w2m   = invert_matching(men_prop_m2w)
    women_prop_w2m = run_gale_shapley(women_utils, men_utils)
    women_prop_m2w = invert_matching(women_prop_w2m)

    men_prop_men_utils     = men_utils[idx, men_prop_m2w]
    women_prop_men_utils   = men_utils[idx, women_prop_m2w]
    men_prop_women_utils   = women_utils[idx, men_prop_w2m]
    women_prop_women_utils = women_utils[idx, women_prop_w2m]

    assert check_stability(men_prop_m2w, men_utils, women_utils)
    assert check_stability(women_prop_m2w, men_utils, women_utils)

    assert (men_prop_men_utils >= women_prop_men_utils).all()
    assert (women_prop_women_utils >= men_prop_women_utils).all()

