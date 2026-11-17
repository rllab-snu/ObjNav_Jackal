"""Create lightweight runtime topology pickles without embedded RGB/depth PNG bytes.

The source pickles remain the canonical image-carrying artifacts used by annotation,
offline VLM scoring, and montage rendering. Runtime navigation needs node positions,
view indices, and metadata only.

Run from the repository root::

    python strip_topology_images.py
"""
import argparse
from pathlib import Path
import pickle

from planning.topology import IMAGE_KEYS, Topology


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "topo_data"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "topo_graph"
TOPOLOGY_NAMES = ("base", "prior1", "prior2", "prior3")


def strip_topology_images(source, destination):
    """Copy one topology pickle while replacing every embedded image with ``None``."""
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination topology paths must differ")

    with source.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise ValueError(f"unsupported topology pickle structure: {source}")

    removed_images = 0
    removed_bytes = 0
    for node in data["nodes"]:
        for view in node.get("views", []):
            images = view.get("images") or {}
            for payload in images.values():
                if payload is not None:
                    removed_images += 1
                    if isinstance(payload, (bytes, bytearray, memoryview)):
                        removed_bytes += len(payload)
            view["images"] = {key: None for key in IMAGE_KEYS}

    # Validate the complete non-image schema before writing the derived artifact.
    Topology.from_dict(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(data, stream, protocol=4)
    return {
        "source": str(source),
        "destination": str(destination),
        "removed_images": removed_images,
        "removed_bytes": removed_bytes,
        "source_bytes": source.stat().st_size,
        "destination_bytes": destination.stat().st_size,
    }


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Strip embedded RGB/depth bytes from runtime topology pickles."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--priors", default="base",
                        help="comma list of topology names to strip (default: base)")
    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for name in [p.strip() for p in args.priors.split(",") if p.strip()]:
        report = strip_topology_images(
            source_dir / f"{name}_topology.pkl",
            output_dir / f"{name}_topology.pkl",
        )
        before = report["source_bytes"]
        after = report["destination_bytes"]
        reduction = 100.0 * (1.0 - after / before)
        print(
            f"[{name}] removed {report['removed_images']} images "
            f"({report['removed_bytes'] / 1e6:.2f} MB): "
            f"{before / 1e6:.2f} MB -> {after / 1e3:.1f} KB "
            f"({reduction:.3f}% smaller)"
        )


if __name__ == "__main__":
    main()
