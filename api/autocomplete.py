"""
Tag Autocomplete API

Provides tag autocomplete functionality with support for:
- Danbooru/e621 tag databases
- Alias searching
- Post count sorting with optional frequency boosting
"""

import csv
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Conditional server import
try:
    from aiohttp import web
    from server import PromptServer

    _HAS_SERVER = True
except ImportError:
    _HAS_SERVER = False
    web = None
    PromptServer = None

# Tag type definitions (for color coding in frontend)
TAG_TYPES = {
    0: "general",  # lightblue/dodgerblue
    1: "artist",  # indianred/firebrick
    3: "copyright",  # violet/darkorchid
    4: "character",  # lightgreen/darkgreen
    5: "meta",  # orange/darkorange
}

# Get the data directory path
DATA_DIR = Path(__file__).parent.parent / "data" / "tags"
WILDCARDS_DIR = Path(__file__).parent.parent / "data" / "wildcards"

# Default tag file to use if not specified (danbooru.csv is default)
DEFAULT_TAG_FILE = "danbooru_e621_merged_2026-03-01_pt20-ia-dd-ed-spc.csv"


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_limit(value, default: int = 20, cap: int = 100) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = default
    return max(0, min(limit, cap))


def _normalize_tag_filename(value, default: Optional[str] = None) -> Optional[str]:
    filename = _coerce_text(value).strip()
    if not filename:
        return default

    filename = filename.replace("\\", "/")
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        return default
    return filename


def _normalize_extra_files(value) -> List[str]:
    if value is None:
        value = ["extra-quality-tags.csv"]
    elif isinstance(value, str):
        value = [value]
    elif not isinstance(value, list):
        value = ["extra-quality-tags.csv"]

    files = []
    for item in value:
        filename = _normalize_tag_filename(item)
        if filename and filename not in files:
            files.append(filename)
    return files


class TagEntry:
    """Represents a single tag entry from the database."""

    __slots__ = ("name", "type", "count", "aliases", "search_text")

    def __init__(self, name: str, tag_type: int, count: int, aliases: List[str]):
        self.name = name
        self.type = tag_type
        self.count = count
        self.aliases = aliases
        # Pre-compute lowercase search text for faster matching
        self.search_text = name.lower()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "count": self.count,
            "aliases": self.aliases,
            "kind": "tag",
        }


class WildcardEntry:
    """Represents a single wildcard file available for completion."""

    __slots__ = ("name", "kind", "search_text", "leaf_name")

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.search_text = name.lower()
        self.leaf_name = name.rsplit("/", 1)[-1]

    def to_dict(self, completion_name: str = "") -> Dict:
        completion_name = completion_name or self.leaf_name
        return {
            "name": self.name,
            "leaf_name": self.leaf_name,
            "kind": self.kind,
            "completion": f"__{completion_name}__",
            "meta": "Wildcard folder" if self.kind == "wildcard_folder" else "Wildcard file",
        }


class TagDatabase:
    """
    In-memory tag database with fast prefix search.

    Tags are loaded lazily on first search to avoid slowing down ComfyUI startup.
    Supports loading multiple tag files and merging results.
    """

    def __init__(self):
        self._tags: List[TagEntry] = []
        self._alias_map: Dict[str, TagEntry] = {}  # alias -> canonical tag
        self._prefix_index: Dict[str, List[Tuple[TagEntry, Optional[str], str]]] = {}
        self._contains_index: Dict[str, List[Tuple[TagEntry, Optional[str], str]]] = {}
        self._loaded = False
        self._current_files: List[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def tag_count(self) -> int:
        return len(self._tags)

    def _add_prefix_index_entry(
        self, text: str, tag: TagEntry, matched_alias: Optional[str] = None
    ) -> None:
        normalized = text.lower().strip()
        if not normalized:
            return

        for prefix_len in range(1, min(3, len(normalized)) + 1):
            prefix = normalized[:prefix_len]
            self._prefix_index.setdefault(prefix, []).append(
                (tag, matched_alias, normalized)
            )

    def _get_contains_index_key(self, query: str) -> str:
        return query[: min(3, len(query))]

    def _get_contains_candidates(
        self, query: str
    ) -> List[Tuple[TagEntry, Optional[str], str]]:
        key = self._get_contains_index_key(query)
        if not key:
            return []

        if key not in self._contains_index:
            candidates: List[Tuple[TagEntry, Optional[str], str]] = []
            for tag in self._tags:
                if key in tag.search_text:
                    candidates.append((tag, None, tag.search_text))
            for alias_lower, tag in self._alias_map.items():
                if key in alias_lower:
                    original_alias = next(
                        (a for a in tag.aliases if a.lower() == alias_lower),
                        alias_lower,
                    )
                    candidates.append((tag, original_alias, alias_lower))
            self._contains_index[key] = candidates

        return self._contains_index[key]

    def load_csv(self, filepath: Path, append: bool = False) -> int:
        """
        Load tags from a CSV file.

        Args:
            filepath: Path to the CSV file
            append: If True, append to existing tags instead of replacing

        Returns:
            Number of tags loaded from this file

        CSV Format: name,type,postCount,"aliases"
        Example: 1girl,0,6008644,"1girls,sole_female"
        """
        if not append:
            self._tags = []
            self._alias_map = {}
            self._prefix_index = {}
            self._current_files = []
        self._contains_index = {}

        if not filepath.exists():
            print(f"[Autocomplete] Tag file not found: {filepath}")
            return 0

        try:
            loaded_count = 0
            existing_tags = {tag.name for tag in self._tags}

            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue

                    name = row[0].strip()
                    if not name or name in existing_tags:
                        continue

                    try:
                        tag_type = int(row[1])
                    except (ValueError, IndexError):
                        tag_type = 0

                    try:
                        count = int(row[2])
                    except (ValueError, IndexError):
                        count = 0

                    # Parse aliases (4th column, comma-separated in quotes)
                    aliases = []
                    if len(row) > 3 and row[3]:
                        aliases = [a.strip() for a in row[3].split(",") if a.strip()]

                    entry = TagEntry(name, tag_type, count, aliases)
                    self._tags.append(entry)
                    existing_tags.add(name)
                    loaded_count += 1
                    self._add_prefix_index_entry(name, entry)

                    # Build alias -> canonical tag mapping
                    for alias in aliases:
                        self._alias_map[alias.lower()] = entry
                        self._add_prefix_index_entry(alias, entry, alias)

            self._loaded = True
            self._current_files.append(filepath.name)
            print(f"[Autocomplete] Loaded {loaded_count} tags from {filepath.name}")
            return loaded_count

        except Exception as e:
            print(f"[Autocomplete] Error loading tag file: {e}")
            return 0

    def load_multiple(self, filepaths: List[Path]) -> int:
        """
        Load tags from multiple CSV files, merging results.

        Args:
            filepaths: List of paths to CSV files

        Returns:
            Total number of tags loaded
        """
        total = 0
        for i, filepath in enumerate(filepaths):
            count = self.load_csv(filepath, append=(i > 0))
            total += count
        return total

    def search(
        self,
        query: str,
        limit: int = 20,
        search_aliases: bool = True,
        contains_fallback: bool = True,
    ) -> List[Dict]:
        """
        Search for tags matching the query.

        Args:
            query: Search query (prefix match)
            limit: Maximum number of results
            search_aliases: Whether to also search aliases

        Returns:
            List of matching tags as dictionaries
        """
        query = _coerce_text(query)
        limit = _coerce_limit(limit, default=20, cap=100)

        if not self._loaded or not query or limit == 0:
            return []

        query_lower = query.lower().strip()
        if not query_lower or len(query_lower) > 255:
            return []

        results: List[Tuple[int, TagEntry, Optional[str]]] = []
        seen_tags = set()

        # Score function: exact match > prefix match > contains match
        def get_score(tag: TagEntry, matched_text: str) -> int:
            if matched_text == query_lower:
                return 3  # Exact match
            elif matched_text.startswith(query_lower):
                return 2  # Prefix match
            else:
                return 1  # Contains match

        def add_result(tag: TagEntry, matched_alias: Optional[str], matched_text: str):
            if tag.name in seen_tags:
                return
            score = get_score(tag, matched_text)
            results.append((score, tag, matched_alias))
            seen_tags.add(tag.name)

        def format_results() -> List[Dict]:
            results.sort(key=lambda x: (-x[0], -x[1].count))
            output = []
            for _, tag, matched_alias in results[:limit]:
                entry = tag.to_dict()
                if matched_alias:
                    entry["matched_alias"] = matched_alias
                output.append(entry)
            return output

        prefix_key = query_lower[: min(3, len(query_lower))]
        for tag, matched_alias, matched_text in self._prefix_index.get(prefix_key, []):
            if matched_alias and not search_aliases:
                continue
            if matched_text == query_lower or matched_text.startswith(query_lower):
                add_result(tag, matched_alias, matched_text)

        if len(results) >= limit:
            return format_results()

        if not contains_fallback:
            return format_results()

        if len(query_lower) >= 2:
            candidates = self._get_contains_candidates(query_lower)
            for tag, matched_alias, matched_text in candidates:
                if matched_alias and not search_aliases:
                    continue
                if query_lower in matched_text:
                    add_result(tag, matched_alias, matched_text)
        else:
            # Backend-only fallback for one-character API searches. The frontend
            # starts at two characters, where the lazy contains index applies.
            for tag in self._tags:
                if tag.name in seen_tags:
                    continue
                if query_lower in tag.search_text:
                    add_result(tag, None, tag.search_text)

            if search_aliases:
                for alias_lower, tag in self._alias_map.items():
                    if tag.name in seen_tags:
                        continue
                    if query_lower in alias_lower:
                        original_alias = next(
                            (a for a in tag.aliases if a.lower() == alias_lower),
                            alias_lower,
                        )
                        add_result(tag, original_alias, alias_lower)

        return format_results()


class WildcardDatabase:
    """In-memory wildcard list with simple fuzzy scoring."""

    def __init__(self, wildcards_dir: Path):
        self._wildcards_dir = wildcards_dir
        self._entries: List[WildcardEntry] = []
        self._entry_map: Dict[str, List[WildcardEntry]] = {}
        self._leaf_counts: Dict[str, int] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def wildcard_count(self) -> int:
        return len(self._entries)

    def load(self) -> int:
        self._entries = []
        self._entry_map = {}
        self._leaf_counts = {}

        if not self._wildcards_dir.exists():
            self._loaded = True
            return 0

        folder_names = set()
        for path in self._wildcards_dir.rglob("*.txt"):
            relative = path.relative_to(self._wildcards_dir).with_suffix("")
            relative_name = relative.as_posix()
            self._entries.append(WildcardEntry(relative_name, "wildcard_file"))

            parts = relative.parts[:-1]
            current_parts = []
            for part in parts:
                current_parts.append(part)
                folder_names.add("/".join(current_parts))

        for folder_name in folder_names:
            self._entries.append(WildcardEntry(folder_name, "wildcard_folder"))

        self._entries.sort(
            key=lambda entry: (
                entry.search_text,
                0 if entry.kind == "wildcard_folder" else 1,
                len(entry.name),
            )
        )
        for entry in self._entries:
            self._entry_map.setdefault(entry.search_text, []).append(entry)
            leaf_key = entry.leaf_name.lower()
            self._leaf_counts[leaf_key] = self._leaf_counts.get(leaf_key, 0) + 1
        self._loaded = True
        return len(self._entries)

    def _entry_to_dict(self, entry: WildcardEntry) -> Dict:
        completion_name = (
            entry.leaf_name
            if self._leaf_counts.get(entry.leaf_name.lower(), 0) == 1
            else entry.name
        )
        return entry.to_dict(completion_name=completion_name)

    def _normalize_query(self, query: str) -> str:
        normalized = str(query or "").strip().lower()
        if normalized.startswith("__"):
            normalized = normalized[2:]
        if normalized.endswith("__"):
            normalized = normalized[:-2]
        return normalized

    def _resolve_entry(self, wildcard_name: str) -> Optional[WildcardEntry]:
        normalized = self._normalize_query(wildcard_name)
        if not normalized:
            return None

        candidates = self._entry_map.get(normalized, [])
        if not candidates:
            return None

        for kind in ("wildcard_folder", "wildcard_file"):
            for entry in candidates:
                if entry.kind == kind:
                    return entry
        return candidates[0]

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        if not self._loaded:
            self.load()

        limit = _coerce_limit(limit, default=20, cap=500)
        normalized = self._normalize_query(query)
        if len(normalized) > 255 or limit == 0:
            return []

        if not normalized:
            return [self._entry_to_dict(entry) for entry in self._entries[:limit]]

        scored: List[Tuple[int, WildcardEntry]] = []
        for entry in self._entries:
            if normalized == entry.search_text:
                score = 3
            elif entry.search_text.startswith(normalized):
                score = 2
            elif normalized in entry.search_text:
                score = 1
            else:
                continue
            scored.append((score, entry))

        scored.sort(
            key=lambda item: (
                -item[0],
                0 if item[1].kind == "wildcard_folder" else 1,
                len(item[1].name),
                item[1].search_text,
            )
        )
        return [self._entry_to_dict(entry) for _, entry in scored[:limit]]

    def get_contents(
        self, wildcard_name: str, content_query: str = "", limit: int = 50
    ) -> List[Dict]:
        if not self._loaded:
            self.load()

        limit = _coerce_limit(limit, default=50, cap=500)
        if limit == 0:
            return []

        entry = self._resolve_entry(wildcard_name)
        if entry is None:
            return []

        if entry.kind == "wildcard_folder":
            prefix = f"{entry.name}/"
            normalized_query = _coerce_text(content_query).strip().lower()
            descendants = []

            for candidate in self._entries:
                if candidate.name == entry.name or not candidate.name.startswith(prefix):
                    continue
                remainder = candidate.name[len(prefix) :]
                if normalized_query and normalized_query not in remainder.lower():
                    continue
                descendants.append(self._entry_to_dict(candidate))
                if len(descendants) >= limit:
                    break

            return descendants

        wildcard_path = self._wildcards_dir.joinpath(*entry.name.split("/")).with_suffix(
            ".txt"
        )
        if not wildcard_path.exists():
            return []

        normalized_query = _coerce_text(content_query).strip().lower()
        results = []
        with open(wildcard_path, "r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if normalized_query and normalized_query not in line.lower():
                    continue
                results.append(
                    {
                        "name": line,
                        "kind": "wildcard_content",
                        "completion": line,
                        "meta": entry.name,
                    }
                )
                if len(results) >= limit:
                    break

        return results


# Global database instance (lazy loaded)
_database = TagDatabase()
_wildcard_database = WildcardDatabase(WILDCARDS_DIR)
_database_load_lock = threading.Lock()
_database_loading = False
_database_load_error: Optional[str] = None
_database_loading_files: List[str] = []


def get_database() -> TagDatabase:
    """Get the global tag database instance."""
    return _database


def get_wildcard_database() -> WildcardDatabase:
    """Get the global wildcard database instance."""
    return _wildcard_database


def _build_tag_file_list(
    tag_file: str = DEFAULT_TAG_FILE,
    extra_files: Optional[List[str]] = None,
) -> List[str]:
    safe_tag_file = _normalize_tag_filename(tag_file, DEFAULT_TAG_FILE)
    files_to_load = [safe_tag_file]
    if extra_files:
        files_to_load.extend(_normalize_extra_files(extra_files))
    return [filename for filename in files_to_load if filename]


def _resolve_tag_filepaths(files_to_load: List[str]) -> List[Path]:
    filepaths = []

    for filename in files_to_load:
        # Try main data directory first
        filepath = DATA_DIR / filename

        # Fall back to reference directory
        if not filepath.exists():
            ref_path = (
                Path(__file__).parent.parent
                / "autocomplete_reference"
                / "a1111-sd-webui-tagcomplete"
                / "tags"
                / filename
            )
            if ref_path.exists():
                filepath = ref_path

        if filepath.exists():
            filepaths.append(filepath)
        else:
            print(f"[Autocomplete] Warning: Tag file not found: {filename}")

    return filepaths


def get_database_status() -> Dict:
    db = get_database()
    return {
        "loaded": db.is_loaded,
        "loading": _database_loading,
        "load_error": _database_load_error,
        "tag_count": db.tag_count,
        "current_files": db._current_files,
        "loading_files": _database_loading_files,
    }


def ensure_database_loaded(
    tag_file: str = DEFAULT_TAG_FILE,
    extra_files: Optional[List[str]] = None,
) -> TagDatabase:
    """
    Ensure the database is loaded, loading it if necessary.

    Args:
        tag_file: Name of the main tag file to load
        extra_files: Optional list of additional tag files to merge (e.g., ["extra-quality-tags.csv"])

    Returns:
        The loaded TagDatabase instance
    """
    global _database_loading, _database_load_error, _database_loading_files
    db = get_database()

    files_to_load = _build_tag_file_list(tag_file, extra_files)

    # Check if we need to reload
    needs_reload = not db.is_loaded or set(db._current_files) != set(files_to_load)
    if not needs_reload:
        if _database_loading and set(_database_loading_files) == set(files_to_load):
            _database_loading = False
            _database_loading_files = []
        return db

    with _database_load_lock:
        needs_reload = not db.is_loaded or set(db._current_files) != set(files_to_load)
        if not needs_reload:
            if _database_loading and set(_database_loading_files) == set(files_to_load):
                _database_loading = False
                _database_loading_files = []
            return db

        _database_loading = True
        _database_load_error = None
        _database_loading_files = files_to_load[:]

        try:
            filepaths = _resolve_tag_filepaths(files_to_load)
            if filepaths:
                db.load_multiple(filepaths)
        except Exception as exc:
            _database_load_error = str(exc)
            raise
        finally:
            _database_loading = False
            _database_loading_files = []

    return db


def start_database_warmup(
    tag_file: str = DEFAULT_TAG_FILE,
    extra_files: Optional[List[str]] = None,
) -> bool:
    global _database_loading, _database_load_error, _database_loading_files
    files_to_load = _build_tag_file_list(tag_file, extra_files)
    db = get_database()
    with _database_load_lock:
        if db.is_loaded and set(db._current_files) == set(files_to_load):
            return False
        if _database_loading:
            return False
        _database_loading = True
        _database_load_error = None
        _database_loading_files = files_to_load[:]

    def warmup_worker():
        try:
            ensure_database_loaded(tag_file, extra_files=extra_files)
        except Exception as exc:
            print(f"[Autocomplete] Warm-up failed: {exc}")

    thread = threading.Thread(
        target=warmup_worker,
        name="A1111AutocompleteWarmup",
        daemon=True,
    )
    thread.start()
    return True


# Register API endpoints (only if server is available)
if _HAS_SERVER and PromptServer:

    @PromptServer.instance.routes.post("/a1111_prompt/autocomplete")
    async def autocomplete_tags(request):
        """
        API endpoint for tag autocomplete.

        Request body:
            {
                "query": "partial_tag_name",
                "limit": 20,
                "tag_file": "danbooru.csv",
                "search_aliases": true
            }

        Response:
            {
                "results": [
                    {
                        "name": "1girl",
                        "type": 0,
                        "count": 6008644,
                        "aliases": ["1girls", "sole_female"],
                        "matched_alias": "sole_female"  // only if matched via alias
                    },
                    ...
                ],
                "tag_count": 100000
            }
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        if not isinstance(data, dict):
            return web.json_response(
                {"error": "Request body must be a JSON object"}, status=400
            )

        query = _coerce_text(data.get("query", ""))
        if len(query) > 255:
            return web.json_response(
                {"results": [], "tag_count": 0, "error": "Query too long"}
            )
        mode = str(data.get("mode", "tag")).lower()
        if mode not in {"tag", "wildcard", "wildcard_contents"}:
            mode = "tag"
        limit_cap = 500 if mode in {"wildcard", "wildcard_contents"} else 100
        limit = _coerce_limit(data.get("limit", 20), default=20, cap=limit_cap)
        tag_file = _normalize_tag_filename(data.get("tag_file"), DEFAULT_TAG_FILE)
        extra_files = _normalize_extra_files(data.get("extra_files"))
        search_aliases = _coerce_bool(data.get("search_aliases", True), default=True)
        contains_fallback = _coerce_bool(
            data.get("contains_fallback", True), default=True
        )

        if mode == "wildcard":
            db = get_wildcard_database()
            results = db.search(query, limit=limit)
            return web.json_response(
                {"results": results, "tag_count": db.wildcard_count, "mode": "wildcard"}
            )

        if mode == "wildcard_contents":
            db = get_wildcard_database()
            wildcard_name = _coerce_text(data.get("wildcard_name", ""))
            content_query = _coerce_text(data.get("content_query", ""))
            results = db.get_contents(
                wildcard_name=wildcard_name,
                content_query=content_query,
                limit=limit,
            )
            return web.json_response(
                {
                    "results": results,
                    "tag_count": len(results),
                    "mode": "wildcard_contents",
                    "wildcard_name": wildcard_name,
                }
            )

        # Ensure database is loaded
        db = ensure_database_loaded(
            tag_file or DEFAULT_TAG_FILE, extra_files=extra_files
        )

        # Perform search
        results = db.search(
            query,
            limit=limit,
            search_aliases=search_aliases,
            contains_fallback=contains_fallback,
        )

        return web.json_response(
            {"results": results, "tag_count": db.tag_count, "mode": "tag"}
        )

    @PromptServer.instance.routes.get("/a1111_prompt/autocomplete/status")
    async def autocomplete_status(request):
        """
        Get the status of the autocomplete database.

        Response:
            {
                "loaded": true,
                "tag_count": 100000,
                "current_file": "danbooru.csv"
            }
        """
        db = get_database()
        return web.json_response(get_database_status())

    @PromptServer.instance.routes.post("/a1111_prompt/autocomplete/warmup")
    async def autocomplete_warmup(request):
        try:
            data = await request.json()
        except Exception:
            data = {}

        if not isinstance(data, dict):
            data = {}

        tag_file = _normalize_tag_filename(data.get("tag_file"), DEFAULT_TAG_FILE)
        extra_files = _normalize_extra_files(data.get("extra_files"))
        started = start_database_warmup(
            tag_file or DEFAULT_TAG_FILE, extra_files=extra_files
        )
        status = get_database_status()
        status["started"] = started
        return web.json_response(status)

    @PromptServer.instance.routes.get("/a1111_prompt/autocomplete/files")
    async def list_tag_files(request):
        """
        List available tag files.

        Response:
            {
                "files": ["danbooru.csv", "e621.csv", ...]
            }
        """
        files = []

        # Check main data directory
        if DATA_DIR.exists():
            files.extend(f.name for f in DATA_DIR.glob("*.csv"))

        # Check reference directory
        ref_dir = (
            Path(__file__).parent.parent
            / "autocomplete_reference"
            / "a1111-sd-webui-tagcomplete"
            / "tags"
        )
        if ref_dir.exists():
            for f in ref_dir.glob("*.csv"):
                if f.name not in files:
                    files.append(f.name)

        files.sort()

        return web.json_response({"files": files})
