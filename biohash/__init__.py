"""Biometric and structural similarity hashing.

Face and document similarity are represented as sparse binary FlyHash tags,
never as embeddings. See flyhash.py for the reasoning.
"""

from biohash.flyhash import TAG_DIM, FlyHash, chance_overlap, minhash, minhash_bands, tag_overlap

__all__ = ["TAG_DIM", "FlyHash", "chance_overlap", "minhash", "minhash_bands", "tag_overlap"]
