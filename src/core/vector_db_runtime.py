from pathlib import Path

from .config import config


def vector_collection_slots() -> tuple[str, str]:
    return (
        config.VECTOR_DB_PRIMARY_COLLECTION_NAME,
        config.VECTOR_DB_STAGING_COLLECTION_NAME,
    )


def backup_collection_name() -> str:
    return config.VECTOR_DB_BACKUP_COLLECTION_NAME


def collection_pointer_path() -> Path:
    return Path(config.VECTOR_DB_COLLECTION_POINTER_PATH)


def resolve_active_collection_name(pointer_path: str | Path | None = None) -> str:
    pointer = Path(pointer_path) if pointer_path is not None else collection_pointer_path()
    default_collection = config.VECTOR_DB_PRIMARY_COLLECTION_NAME
    allowed_collections = set(vector_collection_slots())

    try:
        resolved_name = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return default_collection

    return resolved_name if resolved_name in allowed_collections else default_collection


def write_active_collection_name(
    collection_name: str,
    pointer_path: str | Path | None = None,
) -> Path:
    if collection_name not in set(vector_collection_slots()):
        raise ValueError(f"Unknown vector collection name: {collection_name}")

    pointer = Path(pointer_path) if pointer_path is not None else collection_pointer_path()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"{collection_name}\n", encoding="utf-8")
    return pointer


def inactive_collection_name(active_collection_name: str) -> str:
    primary_collection, staging_collection = vector_collection_slots()
    return staging_collection if active_collection_name == primary_collection else primary_collection
