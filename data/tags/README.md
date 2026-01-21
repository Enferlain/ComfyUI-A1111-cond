# Tag Databases

This directory is intended for tag database files used by the autocomplete feature.

## Supported Formats (Future)

- **Danbooru CSV**: Tag name, post count, category
- **e621 CSV**: Similar format to Danbooru
- **Custom TXT**: Simple list of tags, one per line

## Example Structure

```
tags/
├── danbooru.csv       # Danbooru tag database
├── e621.csv           # e621 tag database
└── custom.txt         # User-defined tags
```

## Notes

- Large tag databases (millions of entries) should be loaded lazily
- Consider using SQLite for very large databases
- Tags should be searchable by prefix for autocomplete
