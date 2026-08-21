"""
Speaker identification: turning anonymous diarization clusters (SPEAKER_00,
SPEAKER_01, ...) into persistent named identities.

This package intentionally does NOT extract voice embeddings itself — that
requires a model choice, a GPU validation pass, and real multi-speaker audio,
none of which exist in this repository yet (see IDENTITY_ARCHITECTURE.md).
What lives here is the embedding-agnostic half of the problem: given *some*
fixed-length vector representing a voice, decide whether it matches a known
person, and given transcript text, decide whether a speaker just told us who
they are.

    self_identification.py  — "I'm Neel" style evidence extraction (text only)
    profile.py               — SpeakerProfile: a person, their embeddings, a centroid
    matcher.py                — cosine similarity + confidence tiers
    store.py                  — local JSON persistence for profiles across jobs
"""
