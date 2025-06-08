# cat_maps.py
"""
Mapping Melee enums → dense embedding indices.
We map each enum's `.name` to a contiguous 0..N-1 index (not the original game IDs).
This is suitable for embedding lookups in our model.
"""
from melee.enums import Stage, Character, Action, ProjectileType

# Dense 0..N-1 index maps for embedding
STAGE_MAP = {s.name: i for i, s in enumerate(Stage)}
CHARACTER_MAP = {c.name: i for i, c in enumerate(Character)}
ACTION_MAP = {a.name: i for i, a in enumerate(Action)}
PROJECTILE_TYPE_MAP = {p.name: i for i, p in enumerate(ProjectileType)}
