"""
Tag Autocomplete API

Provides tag autocomplete functionality with support for:
- Danbooru/e621 tag databases
- Alias searching
- Post count sorting with optional frequency boosting
"""

import csv
import re
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
        self._loaded = False
        self._current_files: List[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def tag_count(self) -> int:
        return len(self._tags)

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
            self._current_files = []

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

                    # Build alias -> canonical tag mapping
                    for alias in aliases:
                        self._alias_map[alias.lower()] = entry

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
        self, query: str, limit: int = 20, search_aliases: bool = True
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
        if not self._loaded or not query:
            return []

        query_lower = query.lower().strip()
        if not query_lower:
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

        # Search by tag name
        for tag in self._tags:
            if query_lower in tag.search_text:
                score = get_score(tag, tag.search_text)
                results.append((score, tag, None))
                seen_tags.add(tag.name)

        # Search by aliases
        if search_aliases:
            for alias_lower, tag in self._alias_map.items():
                if tag.name in seen_tags:
                    continue
                if query_lower in alias_lower:
                    score = get_score(tag, alias_lower)
                    # Find the original case alias
                    original_alias = next(
                        (a for a in tag.aliases if a.lower() == alias_lower),
                        alias_lower,
                    )
                    results.append((score, tag, original_alias))
                    seen_tags.add(tag.name)

        # Sort by: score (desc), then post count (desc)
        results.sort(key=lambda x: (-x[0], -x[1].count))

        # Format results
        output = []
        for _, tag, matched_alias in results[:limit]:
            entry = tag.to_dict()
            if matched_alias:
                entry["matched_alias"] = matched_alias
            output.append(entry)

        return output


# Global database instance (lazy loaded)
_database = TagDatabase()


def get_database() -> TagDatabase:
    """Get the global tag database instance."""
    return _database


def ensure_database_loaded(tag_file: str = "danbooru.csv", extra_files: Optional[List[str]] = None) -> TagDatabase:
    """
    Ensure the database is loaded, loading it if necessary.

    Args:
        tag_file: Name of the main tag file to load
        extra_files: Optional list of additional tag files to merge (e.g., ["extra-quality-tags.csv"])

    Returns:
        The loaded TagDatabase instance
    """
    db = get_database()

    # Build list of files to load
    files_to_load = [tag_file]
    if extra_files:
        files_to_load.extend(extra_files)

    # Check if we need to reload
    needs_reload = not db.is_loaded or set(db._current_files) != set(files_to_load)

    if needs_reload:
        # Find and load all tag files
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
        
        if filepaths:
            db.load_multiple(filepaths)

    return db


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

        query = data.get("query", "")
        limit = min(data.get("limit", 20), 100)  # Cap at 100
        tag_file = data.get("tag_file", "danbooru.csv")
        extra_files = data.get("extra_files", ["extra-quality-tags.csv"])  # Load quality tags by default
        search_aliases = data.get("search_aliases", True)

        # Ensure database is loaded
        db = ensure_database_loaded(tag_file, extra_files=extra_files)

        # Perform search
        results = db.search(query, limit=limit, search_aliases=search_aliases)

        return web.json_response({"results": results, "tag_count": db.tag_count})


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
        return web.json_response(
            {
                "loaded": db.is_loaded,
                "tag_count": db.tag_count,
                "current_files": db._current_files,
            }
        )


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
