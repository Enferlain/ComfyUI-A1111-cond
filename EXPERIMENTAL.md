From hooks.py, needs more testing if it changes the output or not

```py
# NOTE: Pooled output (y) swapping disabled - testing if it improves A1111 parity
# The pooled output might have different timing characteristics that affect
# how the style "settles" in later steps.
# if target_pooled is not None and "y" in c:
#     orig_y = c["y"]
#     if orig_y.shape[-1] == 2816:  # SDXL
#         new_pooled = target_pooled.to(device=device, dtype=dtype).clone()
#         modified_y = orig_y.clone()
#
#         # Only swap positive positions
#         for batch_idx, cond_type in enumerate(cond_or_uncond):
#             if cond_type == 0 and batch_idx < modified_y.shape[0]:
#                 # Replace first 1280 dims with our pooled output
#                 modified_y[batch_idx, :1280] = new_pooled[0]
#
#         c["y"] = modified_y
#         logging.debug(f"  Also swapped pooled output (first 1280 of y)")
```