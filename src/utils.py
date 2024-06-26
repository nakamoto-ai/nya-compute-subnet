import re

from communex.client import CommuneClient
from substrateinterface import Keypair  # type: ignore
from substrateinterface import Keypair

IP_REGEX = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+")


def set_weights(
        score_dict: dict[
            int, float
        ],  # implemented as a float score from 0 to 1, one being the best
        # you can implement your custom logic for scoring
        netuid: int,
        client: CommuneClient,
        key: Keypair
) -> None:
    """
    Set weights for miners based on their scores.

    Args:
        score_dict: A dictionary mapping miner UIDs to their scores.
        netuid: The network UID.
        client: The CommuneX client.
        key: The keypair for signing transactions.
    """

    # Create a new dictionary to store the weighted scores
    weighted_scores: dict[int, int] = {}

    # Calculate the sum of all inverted scores
    scores = sum(score_dict.values())
    # process the scores into weights of type dict[int, int]
    # Iterate over the items in the score_dict
    for uid, score in score_dict.items():
        # Calculate the normalized weight as an integer
        if scores == 0:
            weight = 0
        else:
            weight = int(score * 1000 / scores)

        # Add the weighted score to the new dictionary
        weighted_scores[uid] = weight

    # filter out 0 weights
    weighted_scores = {k: v for k, v in weighted_scores.items() if v != 0}

    uids = list(weighted_scores.keys())
    weights = list(weighted_scores.values())

    client.vote(key=key, uids=uids, weights=weights, netuid=netuid)


def extract_address(string: str):
    """
    Extracts an address from a string.
    """
    return re.search(IP_REGEX, string)


def get_ip_port(modules_addresses: dict[int, str]):
    """
    Get the IP and port information from module addresses.

    Args:
        modules_addresses: A dictionary mapping module IDs to their addresses.

    Returns:
        A dictionary mapping module IDs to their IP and port information.
    """

    filtered_addr = {module_id: extract_address(addr) for module_id, addr in modules_addresses.items()}
    ip_port = {
        module_id: x.group(0).split(":") for module_id, x in filtered_addr.items() if x is not None
    }
    return ip_port
