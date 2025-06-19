"""
Mapping Melee enums → dense embedding indices.
We map each enum's `.value` (raw game ID) to a contiguous 0..N-1 index (suitable for embedding lookups).
"""
from melee.enums import Stage, Character, Action, ProjectileType

# Dense 0..N-1 index maps for embedding, keyed by raw game ID (.value)
STAGE_MAP = {s.value: i for i, s in enumerate(Stage)}
CHARACTER_MAP = {c.value: i for i, c in enumerate(Character)}
ACTION_MAP = {a.value: i for i, a in enumerate(Action)}
PROJECTILE_TYPE_MAP = {p.value: i for i, p in enumerate(ProjectileType)}
