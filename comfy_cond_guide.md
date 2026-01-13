## ComfyUI Conditioning and Text Encoding Pipeline
This codemap traces the complete conditioning pipeline in ComfyUI, from user text prompts through tokenization, weighting, encoding, and application during diffusion sampling. Key highlights include the emphasis parsing system [1c], token weight interpolation [2c], length management for different models [3d], multi-encoder integration [4d], and conditioning concatenation [5c].
### 1. Text Encoding Pipeline from User Input to Model
Traces how user text prompts flow through tokenization, weighting, and encoding to become conditioning for diffusion models
### 1a. CLIP Node Tokenizes Text (`nodes.py:76`)
User-facing node initiates text tokenization
```text
tokens = clip.tokenize(text)
```
### 1b. Tokenizer Called with Options (`sd.py:192`)
SD model delegates to tokenizer with configuration
```text
return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)
```
### 1c. Parse Emphasis Weights (`sd1_clip.py:553`)
Parentheses-based emphasis parsed into weighted tokens
```text
parsed_weights = token_weights(text, 1.0)
```
### 1d. Encode with Scheduling (`nodes.py:77`)
Tokenized text encoded with potential scheduling hooks
```text
return clip.encode_from_tokens_scheduled(tokens)
```
### 2. Token Weight Application and Embedding Interpolation
Shows how weights from emphasis syntax are applied to token embeddings through interpolation
### 2a. Add Empty Token Reference (`sd1_clip.py:41`)
Empty embedding created for weight interpolation baseline
```text
to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
```
### 2b. Get Empty Embedding (`sd1_clip.py:57`)
Reference empty embedding for interpolation
```text
z_empty = out[-1]
```
### 2c. Apply Weight Interpolation (`sd1_clip.py:62`)
Weight applied by interpolating between token and empty embedding
```text
z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
```
### 3. Token Length Management and Batching
Demonstrates how ComfyUI handles token length limits and batches tokens for different model constraints
### 3a. Set Maximum Length (`sd1_clip.py:124`)
Model-specific token limit configured
```text
self.max_length = max_length
```
### 3b. Check Token Limit (`sd1_clip.py:603`)
Batch size checked against maximum length
```text
if len(t_group) + len(batch) > self.max_length - has_end_token:
```
### 3c. Split and Batch Tokens (`sd1_clip.py:607`)
Tokens split across multiple batches when exceeding limit
```text
batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
```
### 3d. T5 Minimum Length (`flux.py:15`)
T5 encoder requires minimum 256 tokens for Flux
```text
max_length=99999999, min_length=256
```
### 4. Multi-Encoder Integration (SDXL/SD3/Flux)
Shows how multiple text encoders are combined for advanced models
### 4a. Encode CLIP-G (`sdxl_clip.py:59`)
Large CLIP encoder processes tokens
```text
g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
```
### 4b. Encode CLIP-L (`sdxl_clip.py:60`)
Base CLIP encoder processes tokens
```text
l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
```
### 4c. Concatenate Encodings (`sdxl_clip.py:62`)
Outputs concatenated for SDXL dual-CLIP
```text
return torch.cat([l_out[:,:cut_to], g_out[:,:cut_to]], dim=-1), g_pooled
```
### 4d. Add T5 to SD3 (`sd3_clip.py:140`)
T5 encoder concatenated for SD3 triple-encoder
```text
out = torch.cat([lg_out, t5_out], dim=-2)
```
### 5. Conditioning Application During Sampling
Traces how encoded conditioning is applied during the diffusion sampling process
### 5a. Concatenate Conditioning (`samplers.py:299`)
Multiple conditioning tensors concatenated
```text
c = cond_cat(c)
```
### 5b. Handle Cross-Attention (`conds.py:156`)
Cross-attention conditioning handles padding
```text
out[k] = conds[0].concat(conds[1:])
```
### 5c. Pad with Repeat (`conds.py:74`)
Conditioning padded to match maximum length
```text
c = c.repeat(1, crossattn_max_len // c.shape[1], 1)
```
### 5d. Final Concatenation (`conds.py:76`)
Final concatenated conditioning returned
```text
return torch.cat(out)
```
### 6. Layer Normalization and Attention Masks
Shows how normalization and masking are applied in the CLIP transformer
### 6a. Layer Norm Definition (`clip_model.py:45`)
Layer normalization defined in each transformer block
```text
self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
```
### 6b. Final Layer Norm (`clip_model.py:108`)
Final normalization applied after transformer
```text
self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
```
### 6c. Prepare Attention Mask (`clip_model.py:118`)
Attention mask prepared for transformer
```text
mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1]))
```
### 6d. Apply Final Norm (`clip_model.py:129`)
Final layer normalization applied to output
```text
x = self.final_layer_norm(x)
```

---

Based on my exploration of the ComfyUI codebase, here are the key files that deal with conditioning, prompting, CLIP text encoding, weighing/emphasis, normalization, and token length management:

## Core Text Encoding and Token Weight Processing

**`comfy/sd1_clip.py`** - This is the primary file for text encoding and token weight management. It contains:
- The `token_weights()` function that parses parentheses-based emphasis syntax (e.g., `(text:1.1)`) [1](#0-0) 
- The `parse_parentheses()` function for parsing nested parentheses [2](#0-1) 
- The `ClipTokenWeightEncoder` class that applies weights to token embeddings by interpolating between weighted and empty embeddings [3](#0-2) 
- The `SDTokenizer` class that handles tokenization with weights, token length management, padding, and batching tokens to CLIP's max_length [4](#0-3) 
- The `SDClipModel` class that processes tokens, handles attention masks, and applies layer normalization [5](#0-4) 

## Main Node Implementation

**`nodes.py`** - Contains the user-facing `CLIPTextEncode` node that calls the CLIP tokenizer and encoder [6](#0-5) 

## Scheduling and Encoding Pipeline

**`comfy/sd.py`** - Handles the encoding pipeline:
- The `encode_from_tokens_scheduled()` method that processes tokens with scheduling and hooks [7](#0-6) 
- The `encode_from_tokens()` method for standard token encoding [8](#0-7) 
- The `tokenize()` method that wraps the tokenizer with options [9](#0-8) 

## Conditioning Data Structures

**`comfy/conds.py`** - Defines conditioning classes for different types of conditioning data:
- `CONDCrossAttn` for cross-attention conditioning with concatenation and padding logic [10](#0-9) 
- `CONDRegular` for standard conditioning processing [11](#0-10) 

## Conditioning Manipulation Nodes

**`nodes.py`** - Contains various conditioning manipulation nodes:
- `ConditioningAverage` for averaging conditioning with weights [12](#0-11) 
- `ConditioningConcat` for concatenating conditioning along the token dimension [13](#0-12) 
- `ConditioningSetArea`, `ConditioningSetMask`, and `ConditioningSetTimestepRange` for regional conditioning [14](#0-13) 

**`node_helpers.py`** - Contains the `conditioning_set_values()` utility function used by conditioning nodes [15](#0-14) 

## Application During Sampling

**`comfy/samplers.py`** - Handles how conditioning is applied during generation:
- The `calc_cond_batch()` function that processes conditioning batches [16](#0-15) 
- The `cond_cat()` function that concatenates conditioning tensors [17](#0-16) 
- Logic for processing conditioning with areas, masks, and multipliers [18](#0-17) 

## Model-Specific Implementations

**`comfy/sdxl_clip.py`** - SDXL-specific implementation that uses dual CLIP encoders (CLIP-L and CLIP-G) and concatenates their outputs [19](#0-18) 

**`comfy/text_encoders/sd3_clip.py`** - SD3 implementation with three text encoders (CLIP-L, CLIP-G, and T5-XXL) [20](#0-19) 

**`comfy/text_encoders/flux.py`** - Flux implementation with CLIP-L and T5-XXL encoders, with T5 having a minimum length of 256 tokens [21](#0-20) 

## Notes

The weighing/emphasis system in ComfyUI uses parentheses syntax where `(text)` increases weight by 1.1x and `(text:number)` sets a specific weight. The tokenizer automatically handles token length management by batching tokens into chunks that fit the model's maximum length (typically 77 for SD1.5/SDXL CLIP, but variable for T5). Weights are applied by interpolating between the weighted embedding and an empty embedding. Different models may have different tokenizer configurations, especially regarding padding behavior and minimum lengths.

### Citations

**File:** comfy/sd1_clip.py (L27-79)
```python
class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            if hasattr(self, "gen_empty_tokens"):
                to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
            else:
                to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.encode(to_encode)
        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1].to(model_management.intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k:k+1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
                extra[k] = v

            r = r + (extra,)
        return r
```

**File:** comfy/sd1_clip.py (L81-298)
```python
class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    LAYERS = [
        "last",
        "pooled",
        "hidden",
        "all"
    ]
    def __init__(self, device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=comfy.clip_model.CLIPTextModel,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, enable_attention_masks=False, zero_out_masked=False,
                 return_projected_pooled=True, return_attention_masks=False, model_options={}):  # clip-vit-base-patch32
        super().__init__()

        if textmodel_json_config is None:
            textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_clip_config.json")
            if "model_name" not in model_options:
                model_options = {**model_options, "model_name": "clip_l"}

        if isinstance(textmodel_json_config, dict):
            config = textmodel_json_config
        else:
            with open(textmodel_json_config) as f:
                config = json.load(f)

        te_model_options = model_options.get("{}_model_config".format(model_options.get("model_name", "")), {})
        for k, v in te_model_options.items():
            config[k] = v

        operations = model_options.get("custom_operations", None)
        quant_config = model_options.get("quantization_metadata", None)

        if operations is None:
            if quant_config is not None:
                operations = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)
                logging.info("Using MixedPrecisionOps for text encoder")
            else:
                operations = comfy.ops.manual_cast

        self.operations = operations
        self.transformer = model_class(config, dtype, device, self.operations)

        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks
        self.execution_device = None

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        self.execution_device = options.get("execution_device", self.execution_device)
        if isinstance(self.layer, list) or self.layer == "all":
            pass
        elif layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]
        self.execution_device = None

    def process_tokens(self, tokens, device):
        end_token = self.special_tokens.get("end", None)
        if end_token is None:
            cmp_token = self.special_tokens.get("pad", -1)
        else:
            cmp_token = end_token

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for x in tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0
            for y in x:
                if isinstance(y, numbers.Integral):
                    if eos:
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)
                    token = int(y)
                    tokens_temp += [token]
                    if not eos and token == cmp_token:
                        if end_token is None:
                            attention_mask[-1] = 0
                        eos = True
                else:
                    other_embeds.append((index, y))
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.transformer.get_input_embeddings()(tokens_embed, out_dtype=torch.float32)
            index = 0
            pad_extra = 0
            embeds_info = []
            for o in other_embeds:
                emb = o[1]
                if torch.is_tensor(emb):
                    emb = {"type": "embedding", "data": emb}

                extra = None
                emb_type = emb.get("type", None)
                if emb_type == "embedding":
                    emb = emb.get("data", None)
                else:
                    if hasattr(self.transformer, "preprocess_embed"):
                        emb, extra = self.transformer.preprocess_embed(emb, device=device)
                    else:
                        emb = None

                if emb is None:
                    index += -1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]
                if emb.shape[-1] == tokens_embed.shape[-1]:
                    tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                    attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                    index += emb_shape - 1
                    embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})
                else:
                    index += -1
                    pad_extra += emb_shape
                    logging.warning("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(emb.shape[-1], tokens_embed.shape[-1]))

            if pad_extra > 0:
                padd_embed = self.transformer.get_input_embeddings()(torch.tensor([[self.special_tokens["pad"]] * pad_extra], device=device, dtype=torch.long), out_dtype=torch.float32)
                tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
                attention_mask = attention_mask + [0] * pad_extra

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info

    def forward(self, tokens):
        if self.execution_device is None:
            device = self.transformer.get_input_embeddings().weight.device
        else:
            device = self.execution_device

        embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)

        attention_mask_model = None
        if self.enable_attention_masks:
            attention_mask_model = attention_mask

        if isinstance(self.layer, list):
            intermediate_output = self.layer
        elif self.layer == "all":
            intermediate_output = "all"
        else:
            intermediate_output = self.layer_idx

        outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32, embeds_info=embeds_info)

        if self.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0:
            return z, pooled_output, extra

        return z, pooled_output

    def encode(self, tokens):
        return self(tokens)

```

**File:** comfy/sd1_clip.py (L302-328)
```python
def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result
```

**File:** comfy/sd1_clip.py (L330-348)
```python
def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out
```

**File:** comfy/sd1_clip.py (L468-639)
```python
class SDTokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None, embedding_size=768, embedding_key='clip_l', tokenizer_class=CLIPTokenizer, has_start_token=True, has_end_token=True, pad_to_max_length=True, min_length=None, pad_token=None, end_token=None, min_padding=None, pad_left=False, tokenizer_data={}, tokenizer_args={}):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, **tokenizer_args)
        self.max_length = tokenizer_data.get("{}_max_length".format(embedding_key), max_length)
        self.min_length = tokenizer_data.get("{}_min_length".format(embedding_key), min_length)
        self.end_token = None
        self.min_padding = min_padding
        self.pad_left = pad_left

        empty = self.tokenizer('')["input_ids"]
        self.tokenizer_adds_end_token = has_end_token
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[0]

        if pad_token is not None:
            self.pad_token = pad_token
        elif pad_with_end:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0

        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name:str):
        '''
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        '''
        split_embed = embedding_name.split()
        embedding_name = split_embed[0]
        leftover = ' '.join(split_embed[1:])
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, "{} {}".format(embedding_name[len(stripped):], leftover))
        return (embed, leftover)

    def pad_tokens(self, tokens, amount):
        if self.pad_left:
            for i in range(amount):
                tokens.insert(0, (self.pad_token, 1.0, 0))
        else:
            tokens.extend([(self.pad_token, 1.0, 0)] * amount)

    def tokenize_with_weights(self, text:str, return_word_ids=False, tokenizer_options={}, **kwargs):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        min_length = tokenizer_options.get("{}_min_length".format(self.embedding_key), self.min_length)
        min_padding = tokenizer_options.get("{}_min_padding".format(self.embedding_key), self.min_padding)

        text = escape_important(text)
        if kwargs.get("disable_weights", False):
            parsed_weights = [(text, 1.0)]
        else:
            parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment)
            split = re.split(' {0}|\n{0}'.format(self.embedding_identifier), to_tokenize)
            to_tokenize = [split[0]]
            for i in range(1, len(split)):
                to_tokenize.append("{}{}".format(self.embedding_identifier, split[i]))

            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                end = 999999999999
                if self.tokenizer_adds_end_token:
                    end = -1
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start:end]])

        #reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length
            if self.end_token is not None:
                has_end_token = 1
            else:
                has_end_token = 0

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - has_end_token:
                    remaining_length = self.max_length - len(batch) - has_end_token
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            self.pad_tokens(batch, remaining_length)
                    #start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        #fill last batch
        if self.end_token is not None:
            batch.append((self.end_token, 1.0, 0))
        if min_padding is not None:
            self.pad_tokens(batch, min_padding)
        if self.pad_to_max_length and len(batch) < self.max_length:
            self.pad_tokens(batch, self.max_length - len(batch))
        if min_length is not None and len(batch) < min_length:
            self.pad_tokens(batch, min_length - len(batch))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

        return batched_tokens
```

**File:** nodes.py (L57-78)
```python
class CLIPTextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens), )

```

**File:** nodes.py (L92-128)
```python
class ConditioningAverage :
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ), "conditioning_from": ("CONDITIONING", ),
                              "conditioning_to_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addWeighted"

    CATEGORY = "conditioning"

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
        out = []

        if len(conditioning_from) > 1:
            logging.warning("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )
```

**File:** nodes.py (L130-156)
```python
class ConditioningConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_to": ("CONDITIONING",),
            "conditioning_from": ("CONDITIONING",),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"

    CATEGORY = "conditioning"

    def concat(self, conditioning_to, conditioning_from):
        out = []

        if len(conditioning_from) > 1:
            logging.warning("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from),1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out, )
```

**File:** nodes.py (L158-279)
```python
class ConditioningSetArea:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"area": (height // 8, width // 8, y // 8, x // 8),
                                                                "strength": strength,
                                                                "set_area_to_bounds": False})
        return (c, )

class ConditioningSetAreaPercentage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                              "height": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                              "x": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                              "y": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"area": ("percentage", height, width, y, x),
                                                                "strength": strength,
                                                                "set_area_to_bounds": False})
        return (c, )

class ConditioningSetAreaStrength:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, strength):
        c = node_helpers.conditioning_set_values(conditioning, {"strength": strength})
        return (c, )


class ConditioningSetMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "mask": ("MASK", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "set_cond_area": (["default", "mask bounds"],),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, mask, set_cond_area, strength):
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        c = node_helpers.conditioning_set_values(conditioning, {"mask": mask,
                                                                "set_area_to_bounds": set_area_to_bounds,
                                                                "mask_strength": strength})
        return (c, )

class ConditioningZeroOut:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"

    CATEGORY = "advanced/conditioning"

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c, )

class ConditioningSetTimestepRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_range"

    CATEGORY = "advanced/conditioning"

    def set_range(self, conditioning, start, end):
        c = node_helpers.conditioning_set_values(conditioning, {"start_percent": start,
                                                                "end_percent": end})
        return (c, )
```

**File:** comfy/sd.py (L186-192)
```python
    def tokenize(self, text, return_word_ids=False, **kwargs):
        tokenizer_options = kwargs.get("tokenizer_options", {})
        if len(self.tokenizer_options) > 0:
            tokenizer_options = {**self.tokenizer_options, **tokenizer_options}
        if len(tokenizer_options) > 0:
            kwargs["tokenizer_options"] = tokenizer_options
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)
```

**File:** comfy/sd.py (L199-256)
```python
    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict: dict[str]={}, show_pbar=True):
        all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
        all_hooks = self.patcher.forced_hooks
        if all_hooks is None or not self.use_clip_schedule:
            # if no hooks or shouldn't use clip schedule, do unscheduled encode_from_tokens and perform add_dict
            return_pooled = "unprojected" if unprojected else True
            pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)
            cond = pooled_dict.pop("cond")
            # add/update any keys with the provided add_dict
            pooled_dict.update(add_dict)
            all_cond_pooled.append([cond, pooled_dict])
        else:
            scheduled_keyframes = all_hooks.get_hooks_for_clip_schedule()

            self.cond_stage_model.reset_clip_options()
            if self.layer_idx is not None:
                self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
            if unprojected:
                self.cond_stage_model.set_clip_options({"projected_pooled": False})

            self.load_model()
            self.cond_stage_model.set_clip_options({"execution_device": self.patcher.load_device})
            all_hooks.reset()
            self.patcher.patch_hooks(None)
            if show_pbar:
                pbar = ProgressBar(len(scheduled_keyframes))

            for scheduled_opts in scheduled_keyframes:
                t_range = scheduled_opts[0]
                # don't bother encoding any conds outside of start_percent and end_percent bounds
                if "start_percent" in add_dict:
                    if t_range[1] < add_dict["start_percent"]:
                        continue
                if "end_percent" in add_dict:
                    if t_range[0] > add_dict["end_percent"]:
                        continue
                hooks_keyframes = scheduled_opts[1]
                for hook, keyframe in hooks_keyframes:
                    hook.hook_keyframe._current_keyframe = keyframe
                # apply appropriate hooks with values that match new hook_keyframe
                self.patcher.patch_hooks(all_hooks)
                # perform encoding as normal
                o = self.cond_stage_model.encode_token_weights(tokens)
                cond, pooled = o[:2]
                pooled_dict = {"pooled_output": pooled}
                # add clip_start_percent and clip_end_percent in pooled
                pooled_dict["clip_start_percent"] = t_range[0]
                pooled_dict["clip_end_percent"] = t_range[1]
                # add/update any keys with the provided add_dict
                pooled_dict.update(add_dict)
                # add hooks stored on clip
                self.add_hooks_to_dict(pooled_dict)
                all_cond_pooled.append([cond, pooled_dict])
                if show_pbar:
                    pbar.update(1)
                model_management.throw_exception_if_processing_interrupted()
            all_hooks.reset()
        return all_cond_pooled
```

**File:** comfy/sd.py (L258-281)
```python
    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model()
        self.cond_stage_model.set_clip_options({"execution_device": self.patcher.load_device})
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            self.add_hooks_to_dict(out)
            return out

        if return_pooled:
            return cond, pooled
        return cond
```

**File:** comfy/conds.py (L7-33)
```python
class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(comfy.utils.repeat_to_batch_size(self.cond, batch_size))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        if self.cond.device != other.cond.device:
            logging.warning("WARNING: conds not on same device, skipping concat.")
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)

    def size(self):
        return list(self.cond.size())

```

**File:** comfy/conds.py (L46-77)
```python
class CONDCrossAttn(CONDRegular):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]: #these 2 cases should not happen
                return False

            mult_min = math.lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4: #arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                return False
        if self.cond.device != other.cond.device:
            logging.warning("WARNING: conds not on same device: skipping concat.")
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = math.lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1) #padding with repeat doesn't change result
            out.append(c)
        return torch.cat(out)

```

**File:** node_helpers.py (L8-22)
```python
def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c
```

**File:** comfy/samplers.py (L145-158)
```python
def cond_cat(c_list):
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out
```

**File:** comfy/samplers.py (L203-214)
```python
def calc_cond_batch(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options: dict[str]):
    handler: comfy.context_windows.ContextHandlerABC = model_options.get("context_handler", None)
    if handler is None or not handler.should_use_context(model, conds, x_in, timestep, model_options):
        return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
    return handler.execute(_calc_cond_batch_outer, model, conds, x_in, timestep, model_options)

def _calc_cond_batch_outer(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _calc_cond_batch,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True)
    )
    return executor.execute(model, conds, x_in, timestep, model_options)
```

**File:** comfy/samplers.py (L216-300)
```python
def _calc_cond_batch(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    out_conds = []
    out_counts = []
    # separate conds by matching hooks
    hooked_to_run: dict[comfy.hooks.HookGroup,list[tuple[tuple,int]]] = {}
    default_conds = []
    has_default_conds = False

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        default_c = []
        if cond is not None:
            for x in cond:
                if 'default' in x:
                    default_c.append(x)
                    has_default_conds = True
                    continue
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                if p.hooks is not None:
                    model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
                hooked_to_run.setdefault(p.hooks, list())
                hooked_to_run[p.hooks] += [(p, i)]
        default_conds.append(default_c)

    if has_default_conds:
        finalize_default_conds(model, hooked_to_run, default_conds, x_in, timestep, model_options)

    model.current_patcher.prepare_state(timestep)

    # run every hooked_to_run separately
    for hooks, to_run in hooked_to_run.items():
        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            free_memory = model_management.get_free_memory(x_in.device)
            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
                cond_shapes = collections.defaultdict(list)
                for tt in batch_amount:
                    cond = {k: v.size() for k, v in to_run[tt][0].conditioning.items()}
                    for k, v in to_run[tt][0].conditioning.items():
                        cond_shapes[k].append(v.size())

                if model.memory_required(input_shape, cond_shapes=cond_shapes) * 1.5 < free_memory:
                    to_batch = batch_amount
                    break

            input_x = []
            mult = []
            c = []
            cond_or_uncond = []
            uuids = []
            area = []
            control = None
            patches = None
            for x in to_batch:
                o = to_run.pop(x)
                p = o[0]
                input_x.append(p.input_x)
                mult.append(p.mult)
                c.append(p.conditioning)
                area.append(p.area)
                cond_or_uncond.append(o[1])
                uuids.append(p.uuid)
                control = p.control
                patches = p.patches

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)
```

**File:** comfy/sdxl_clip.py (L41-68)
```python
class SDXLClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.clip_l = sd1_clip.SDClipModel(layer="hidden", layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False, model_options=model_options)
        self.clip_g = SDXLClipG(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = set([dtype])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        cut_to = min(l_out.shape[1], g_out.shape[1])
        return torch.cat([l_out[:,:cut_to], g_out[:,:cut_to]], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)
```

**File:** comfy/text_encoders/sd3_clip.py (L41-58)
```python
class SD3Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.clip_g = sdxl_clip.SDXLClipGTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

    def state_dict(self):
        return {}
```

**File:** comfy/text_encoders/flux.py (L12-33)
```python
class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256, tokenizer_data=tokenizer_data)


class FluxTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}
```
