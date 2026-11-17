"""Topology container: a set of nodes (edges NOT implemented yet).

A `Topology` holds `TopoNode` instances; each node holds its world position (x, z), a room
type, and 12 direction `NodeView`s (dir_00 = yaw 0, then +30 deg steps). Each view carries the
4 raw images (main/sub x rgb/depth) as the ORIGINAL PNG bytes, so a saved topology is fully
self-contained (no dependency on the on-disk capture folder).

FORWARD-COMPATIBLE by design: every level (Topology / TopoNode / NodeView) has an open `meta`
dict for future fields (e.g. a per-node object-category list), and (de)serialisation goes
through to_dict / from_dict which
  * read known fields with defaults (missing field -> default, so old files still load), and
  * stash any UNKNOWN keys into `meta` (so a new field written by newer code survives a
    round-trip through older code).
So adding node info later needs no format change and never breaks existing topology files.

The canonical build is image-carrying (edges are not modelled). Lightweight runtime copies may
set all image slots to ``None`` while preserving node/view metadata; navigation does not read the
embedded images.
"""
import pickle

# image slots each view carries (main + optional sub camera, rgb + depth)
IMAGE_KEYS = ("main_rgb", "main_depth", "sub_rgb", "sub_depth")


class NodeView:
    """One of a node's 12 directions. `index` 0..11 -> yaw = index*30 deg from the node's
    start (dir_00 = yaw 0). `images` maps an IMAGE_KEYS slot to the raw PNG bytes (or None).
    Extra per-view info goes in `meta`."""

    def __init__(self, index, images=None, meta=None):
        self.index = int(index)
        self.images = {k: (images or {}).get(k) for k in IMAGE_KEYS}
        self.meta = dict(meta or {})

    @property
    def nominal_deg(self):
        return (self.index * 30) % 360

    def raw(self, key):
        """Raw PNG bytes for a slot (main_rgb / main_depth / sub_rgb / sub_depth), or None."""
        return self.images.get(key)

    def image(self, key):
        """Decoded image for a slot: rgb -> HxWx3 uint8 RGB, depth -> HxW uint16 (raw). None if
        the slot is empty. Decoding is lazy (needs cv2 + numpy)."""
        b = self.images.get(key)
        if b is None:
            return None
        import cv2, numpy as np
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is not None and key.endswith("_rgb") and arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)   # PNGs were written BGR -> back to RGB
        return arr

    def to_dict(self):
        return {"index": self.index, "images": dict(self.images), "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d):
        known = {"index", "images", "meta"}
        meta = dict(d.get("meta", {}))
        meta.update({k: v for k, v in d.items() if k not in known})   # keep unknown keys
        return cls(d["index"], d.get("images", {}), meta)


class TopoNode:
    """A topology node: `id`, world position (`x`, `z`), `room_type`, and up to 12 ordered
    `views` (index 0..11). `meta` is an open dict for future per-node info -- e.g.
    `node.meta['object_categories'] = [...]` -- added later without changing the format."""

    def __init__(self, node_id, x, z, room_type=None, views=None, meta=None):
        self.id = node_id
        self.x = float(x)
        self.z = float(z)
        self.room_type = room_type
        self.views = list(views or [])
        self.meta = dict(meta or {})

    @property
    def xz(self):
        return (self.x, self.z)

    def add_view(self, view):
        self.views.append(view)
        return view

    def view(self, index):
        """The view at a given direction index (0..11), or None."""
        for v in self.views:
            if v.index == index:
                return v
        return None

    def to_dict(self):
        return {"id": self.id, "x": self.x, "z": self.z, "room_type": self.room_type,
                "views": [v.to_dict() for v in self.views], "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d):
        known = {"id", "x", "z", "room_type", "views", "meta"}
        meta = dict(d.get("meta", {}))
        meta.update({k: v for k, v in d.items() if k not in known})   # keep unknown keys
        return cls(d["id"], d["x"], d["z"], d.get("room_type"),
                   [NodeView.from_dict(v) for v in d.get("views", [])], meta)


class Topology:
    """The whole topology: a set of `TopoNode`s (edges not implemented yet). Build it by adding
    node instances via `add_node`. `name` labels it (e.g. the prior), `meta` holds
    topology-level info. save()/load() are self-contained pickles (raw PNG bytes included)."""

    VERSION = 1

    def __init__(self, name=None, meta=None):
        self.name = name
        self.nodes = []                     # ordered list[TopoNode]
        self.meta = dict(meta or {})

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def add_node(self, node):
        """Add a TopoNode instance (the intended way to build a topology)."""
        if not isinstance(node, TopoNode):
            raise TypeError("Topology.add_node expects a TopoNode instance")
        self.nodes.append(node)
        return node

    def node(self, node_id):
        """The node with this id, or None."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def to_dict(self):
        return {"version": self.VERSION, "name": self.name,
                "nodes": [n.to_dict() for n in self.nodes], "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d):
        known = {"version", "name", "nodes", "meta"}
        meta = dict(d.get("meta", {}))
        meta.update({k: v for k, v in d.items() if k not in known})   # keep unknown keys
        t = cls(d.get("name"), meta)
        for nd in d.get("nodes", []):
            t.add_node(TopoNode.from_dict(nd))
        return t

    # -- persistence: plain pickle (PNG bytes are already compressed; gzip would not help) --
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f, protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls.from_dict(pickle.load(f))
