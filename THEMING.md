# Theming Support

The A1111 Prompt Node autocomplete respects ComfyUI's theme system and will automatically adapt to different color schemes.

## CSS Variables Used

The A1111 Prompt Node UI components use the following ComfyUI theme variables:

### Autocomplete Popup

**Background & Borders:**
- `--comfy-menu-bg` - Popup background color (fallback: `#353535`)
- `--border-color` - Border and separator colors (fallback: `#555`)
- `--comfy-input-bg` - Scrollbar track background (fallback: `#222`)

**Text Colors:**
- `--fg-color` - Primary text color (fallback: `#fff`)
- `--descrip-text` - Secondary text (post counts) (fallback: `#888`)

**Interactive Elements:**
- `--interface-panel-hover-surface` - Selected item background (fallback: `--content-hover-bg`, then `#404040`)

### Token Counter Tooltip

**Background & Borders:**
- `--comfy-menu-bg` - Tooltip background color (fallback: `#353535`)
- `--border-color` - Border and divider colors (fallback: `#555`)
- `--comfy-input-bg` - Token display and input ID backgrounds (fallback: `#222`)

**Text Colors:**
- `--fg-color` - Primary text color (fallback: `#eee`)
- `--descrip-text` - Secondary text (fallback: `#888`)
- `--input-text` - Button text color (fallback: `#aaa`)

**Interactive Elements:**
- `--interface-panel-hover-surface` - Button hover background (fallback: `--content-hover-bg`, then `#404040`)

## Tag Type Colors

Tag colors are fixed to match Danbooru/e621 conventions and remain consistent across themes:

| Type      | Color       | Hex       |
|-----------|-------------|-----------|
| General   | Light Blue  | `#87ceeb` |
| Artist    | Red         | `#cd5c5c` |
| Copyright | Violet      | `#ee82ee` |
| Character | Green       | `#90ee90` |
| Meta      | Orange      | `#ffa500` |

These colors are intentionally not theme-aware to maintain consistency with the tag database conventions.

## Frequency Badge

The frequency indicator (★) uses a fixed blue color (`#4a9eff`) with semi-transparent background for visibility across all themes.

## Custom Theming

If you want to customize the appearance beyond the default theme variables, you can add custom CSS to your `user.css` file:

```css
/* Custom autocomplete styling */
.a1111-autocomplete-popup {
  background: #1a1a1a !important;
  border: 2px solid #00ff00 !important;
}

.autocomplete-item {
  padding: 10px !important;
}

.autocomplete-item:hover {
  background: #2a2a2a !important;
}

/* Custom token tooltip styling */
.a1111-token-tooltip {
  background: #1a1a1a !important;
  border: 2px solid #00ff00 !important;
}
```

### user.css Location

Place your `user.css` file in:
- `ComfyUI/user/default/user.css` (default user)
- Or your custom user directory

## Testing with Different Themes

To test the autocomplete with different themes:

1. Open ComfyUI Settings (gear icon)
2. Go to "Appearance" section
3. Select different color palettes
4. The autocomplete will automatically adapt

## Supported Themes

The autocomplete works with:
- ✅ Dark (Default)
- ✅ Light themes
- ✅ Custom themes
- ✅ Community themes

All themes that properly define ComfyUI's CSS variables will work correctly.

## Fallback Behavior

If a theme doesn't define certain CSS variables, the autocomplete will use sensible fallback values that work well with dark themes (the most common ComfyUI theme style).

## Known Limitations

1. **Tag colors are fixed**: Tag type colors don't change with theme to maintain consistency with Danbooru/e621 conventions
2. **Frequency badge color**: The ★ badge uses a fixed blue color for visibility
3. **Shadow effects**: Box shadows use fixed rgba values for consistency

These limitations are intentional design choices to maintain visual consistency and readability across all themes.

---

_Last updated: 2026-01-22_
