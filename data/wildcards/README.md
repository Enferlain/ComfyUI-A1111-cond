# Wildcards

This directory is intended for wildcard text files used by the wildcard expansion feature.

## Format

Each `.txt` file contains one option per line. The filename (without extension) becomes the wildcard name.

### Example

`colors.txt`:

```
red
blue
green
yellow
purple
```

In prompts, use `__colors__` to randomly select one of these options.

## Nested Wildcards

Wildcards can reference other wildcards:

`outfits.txt`:

```
__colors__ dress
__colors__ shirt
black leather jacket
```

## Directory Structure

```
wildcards/
├── colors.txt
├── artists.txt
├── styles.txt
└── characters/
    ├── anime.txt
    └── cartoon.txt
```

Nested directories use dot notation: `__characters.anime__`
