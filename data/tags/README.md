### Autocomplete feature based on [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)

# Tag Database Files

This directory contains CSV files with tag databases for autocomplete functionality.

## Available Tag Files

The autocomplete system will automatically look for tag files in:
1. This directory (`data/tags/`)

## Currently Available Files

- **danbooru.csv** - Main Danbooru tag database (~140k tags)
- **e621.csv** - E621 tag database (furry-focused)
- **extra-quality-tags.csv** - Quality and style tags
- **EnglishDictionary.csv** - Common English words

## CSV Format

Tag files use the following CSV format:
```
name,type,postCount,"aliases"
```

Example:
```csv
1girl,0,6008644,"1girls,sole_female"
solo,0,3426446,"female_solo,solo_female"
```

### Tag Types

| Type | Category  | Color       |
|------|-----------|-------------|
| 0    | General   | Light Blue  |
| 1    | Artist    | Red         |
| 3    | Copyright | Violet      |
| 4    | Character | Green       |
| 5    | Meta      | Orange      |

## Adding Custom Tags

To add your own tags:

1. Create a CSV file following the format above
2. Place it in this directory
3. The autocomplete will automatically detect it
4. Use the API endpoint `/a1111_prompt/autocomplete/files` to see available files

## Changing the Tag Database

By default, the autocomplete uses `danbooru.csv`. To use a different tag file:

### Method 1: Copy to data/tags/ (Recommended)

Copy your preferred CSV file to the `data/tags/` directory and rename it to `danbooru.csv`:

```bash
# Example: Use the merged Danbooru + e621 database
cp autocomplete_reference/a1111-sd-webui-tagcomplete/tags/danbooru_e621_merged.csv data/tags/danbooru.csv
```

The autocomplete will automatically use this file on next load.

### Method 2: Modify the Backend (Advanced)

Edit `api/autocomplete.py` and change the default tag file:

```python
def ensure_database_loaded(tag_file: str = "danbooru_e621_merged.csv") -> TagDatabase:
    # Change the default from "danbooru.csv" to your preferred file
```

### Available Tag Files

Located in `autocomplete_reference/a1111-sd-webui-tagcomplete/tags/`:

- `danbooru.csv` - Standard Danbooru tags (~140k tags)
- `danbooru_e621_merged.csv` - Combined Danbooru + e621 tags
- `danbooru_e621_merged_2025-12-07_pt20-ia-dd-ed-spc.csv` - Enhanced merged database
- `e621.csv` - E621 tags only (furry-focused)
- `e621_sfw.csv` - SFW e621 tags only
- `extra-quality-tags.csv` - Quality and style tags
- `EnglishDictionary.csv` - Common English words

### Reload After Changing

After changing the tag file, restart ComfyUI or reload the page for changes to take effect.

## Usage

The autocomplete system will:
- Load tags on first search (lazy loading)
- Search by tag name and aliases
- Sort results by relevance and post count
- Show color-coded results by tag type
