# Laya notes (Step 1)

Source of truth for how Laya behaves. Every claim below points at a file and line in Laya at a pinned commit. When this file and the Zet brief disagree, this file wins, and the difference is listed under [Differences from the brief](#differences-from-the-brief).

**Pinned source:** `github.com/NandhaKishorM/laya` @ `9d955671415fc19f069b9cc998928075c1f255ec` (committed 2026-09-28). PyPI `laya` 0.3.21. Line numbers refer to that commit.

**Not verified here:** Hugging Face downloads are blocked in the sandbox these notes were written in. No weights were run, and statements about Hub repos come from their model cards. Step 2's equivalence test is where they get verified.

---

## The answer: per-option probabilities for all three types without PyTorch

Laya's published Python package cannot do this as shipped (see [Why the package can't be used directly](#why-the-laya-package-cant-be-used-directly)). Zet does it with ports of five Laya functions plus ONNX Runtime:

1. **Validate and normalise each question.** Port `Agent._check_question` (`laya/agent.py:618`) and `ONNXAgent._to_internal` (`laya/onnx_agent.py:217–231`). The latter turns a choice list into a dict, lower-cases noul keys, and JSON-encodes non-string instructions.
2. **Tokenise the state once.** Serialise it (`serialize_state`, `laya/common.py:74`: strings pass through, dicts and lists become `json.dumps(..., ensure_ascii=False)`), replace the mask token with a space, and encode with no special tokens (`laya/onnx_agent.py:533–537`).
3. **Build one input row per (state, question).** Use `build_sequence` (`laya/common.py:135–208`), which returns `ids` and `markers`, the positions of each option's `[MASK]`. Details under [Input format](#input-format).
4. **Collate the rows into padded int64 arrays.** Port `collate_items` (`laya/common.py:527–569`) with numpy in place of torch: `input_ids` (pad id), `attention_mask`, `marker_pos`, `marker_mask` (bool), `qtype` (0 choice, 1 score, 2 noul; `laya/common.py:17`).
5. **Run ONNX Runtime.** The single-graph and split formats differ in how, and both are described under [ONNX formats](#onnx-formats). The result is `logits[row, :k]`, where `k = len(markers)`.
6. **Apply the temperature.** `p = softmax(logits[row, :k] / T)`, where `T` is picked by `temp_bucket(qtype, k)`, falling back to the per-type temperature, both clamped to [0.5, 5.0] (`laya/onnx_agent.py:567–578`, `laya/common.py:499–520`).
7. **Map `p` to options.**
   - **choice:** `p[i]` belongs to the i-th key of `criteria` in insertion order (`laya/onnx_agent.py:588–593`).
   - **score:** `p[i]` belongs to level `i`, keyed `"0"…"k-1"` (`laya/onnx_agent.py:597–603`).
   - **noul:** option order is always `[false, true]`, so `p[0]` is false and `p[1]` is true (`laya/common.py:107`, `laya/common.py:127–133`, `laya/onnx_agent.py:611`).

Zet computes the softmax itself, in float64, from the raw logits. It does not reuse Laya's decoded answer dict, because that dict rounds every probability to 4 decimals (see [Output format](#output-format)).

---

## Checkpoints

From the `laya/router.py` docstring (`laya/router.py:1–27`, `47–60`):

| name | Hub location | encoder | max tokens |
|---|---|---|---|
| english | `convaiinnovations/laya` (root) | ModernBERT-large, 421M | 512 |
| multilingual | `convaiinnovations/laya`, subfolder `multilingual` (also `convaiinnovations/laya-multilingual`) | mmBERT-base, 322M | 1024 |
| typed-decisions | subfolder `typed-decisions` | ModernBERT-large, fine-tuned on 4 synthetic workflows | 1024 |

Each checkpoint ships `rl_agent_config.json`, which holds `max_len`, `head_max_len` (default 192), `temperature` (3 floats, one per question type) and `temperature_by_options` (bucket → float). `typed-decisions` is never chosen automatically by Laya (`laya/router.py:24–27`). Zet 0.1 exposes it only if a task names it explicitly.

---

## Input format

`build_sequence` (`laya/common.py:135–208`) builds:

```
[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
```

- **Options are rendered by `render_options`** (`laya/common.py:106–133`):
  - choice: `"label"`, or `"label: description"` when the description is non-empty.
  - score: `"level i: text"`.
  - noul: `"false: …"` then `"true: …"`, with default descriptions *"no, the statement does not hold"* / *"yes, the statement holds"*. Custom display labels come from `labels` (`laya/common.py:92–104`).
- **Each option gets a leading space before tokenising** and is capped at 48 tokens (`laya/common.py:165–171`).
- **Head budget:** if the question and options exceed `head_max_len` (so fewer than 16 tokens of room remain), each option is cut to `max(4, (head_max_len−16)//n)` tokens. The instruction text keeps at least 8 tokens (`laya/common.py:173–179`).
- **State truncation:** the state fills the remaining room up to `max_len`. Strings and dicts are truncated from the right, lists (conversation turns) from the left (`laya/common.py:186–191`, `laya/onnx_agent.py:531`).
- **Markers past `max_len` are dropped.** `_encode_state` raises if the marker count no longer matches the option count (`laya/onnx_agent.py:545–558`).
- **Option collisions:** two options can end up with identical token spans. The collision is reported in `usage.options` (`laya/common.py:193–208`, `210–223`). Zet should surface this as a warning on the task.
- **Tokenizer:** Laya loads it with `transformers.AutoTokenizer` (`laya/onnx_agent.py:97`, `151`). Zet uses `tokenizers.Tokenizer.from_file("tokenizer.json")` with `add_special_tokens=False`, and reads the special ids (`[CLS]`, `[SEP]`, `[MASK]`, pad) from the tokenizer.
  - The Layar Ruby port reports exact token-id parity with the Python package using `tokenizer.json`, so this is feasible. Step 2 must still test it: tokenise every fixture through both paths and compare ids.
  - `truncation=True, max_length=48` on an option equals slicing the first 48 ids (`laya/common.py:162–164`).

**One row per question, not one pass per state.** The state is tokenised once, but the encoder runs over every (state, question) row, because the question text sits inside the sequence. A state with 4 questions is 4 encoder rows in one batched call.

---

## ONNX formats

### Format A: single graph (`laya.onnx` / `model.onnx`)

- **Used by** Laya's Python `ONNXAgent` (`laya/onnx_agent.py:47`, `174–175`).
- **Inputs:** `input_ids` int64, `attention_mask` int64, `marker_pos` int64, `marker_mask` bool, `qtype` int64 (`laya/onnx_agent.py:661–667`).
- **Outputs:** `logits`, `act_logits` (`laya/onnx_agent.py:670`).
- **Export:** the in-repo exporter is `scripts/export_onnx.py` (pulled in by the `laya[onnx]` extra, `pyproject.toml:55–57`).

### Format B: split (`encoder.onnx` + `head.onnx`)

- **Produced by** `laya-ts/scripts/export_onnx.py`.
- **encoder.onnx:** `(input_ids, attention_mask)` → `last_hidden_state` (`export_onnx.py:96–103`, `166–175`).
- **head.onnx:** `(hidden_states, marker_pos, marker_mask, qtype, attention_mask)` → `(logits, act_logits)` (`export_onnx.py:105–128`, `176–184`).
  - The head takes `attention_mask` because the head transformer must not attend to padding (`export_onnx.py:113–119`, matching `laya/common.py:318–326`).
  - Older conversions may lack this input. Zet checks `session.get_inputs()` and refuses a head without `attention_mask`, because unequal-length batch mates would then corrupt each other.
- **`qtype` shape differs:** the head reads `qtype.squeeze(-1)`, so it expects shape `[n, 1]` (the TS port feeds `[n, 1]`, `laya-ts/src/providers.ts:153`). The single graph takes `[n]`.
- **Export check:** it verifies against torch within 1e-4 on logits and act-probabilities, at sequence lengths 16 and 512 (`export_onnx.py:10–17`, `222–242`).
- **Consumers:** the TS port loads it for Node and Web (`laya-ts/src/providers.ts:297`, `428–445`, `feedHead` at `118–157`).

### Existing conversions on the Hub

| repo | format | precision | checkpoints | notes |
|---|---|---|---|---|
| `harshpreet931/cut-laya-onnx` | split | 8-bit weights, fp16 embeddings | english | same top label as PyTorch on 57 lines, probabilities within 0.021 |
| `archevel/laya-web` | split | as above, fp32 embeddings | english | derived from the row above |
| `distinctinteractive/laya-onnx` (**offline since 2026-09-29**) | single (`model.onnx` + `.data`) | fp32 | english + multilingual | pinned source revision `55cf4c4e…`; parity with PyTorch within 2e-6 including a 931-token input; Apache-2.0 with NOTICE |
| `inferenceprince/laya-onnx` | single | fp16 | english | |
| `soyelmismo/laya-multilingual-onnx` | single | int8 `model.onnx`, fp32 `model-fp32.onnx` | multilingual | **Zet's current multilingual default** (fp32 file, commit 0966c4f) |
| `aaronalexS/daylens-laya-onnx` | single | fp32 | typed-decisions | |

Consequences for Zet:

- **No full-precision split export of the multilingual checkpoint exists.** Your plan to run `export_onnx.py --subfolder multilingual` in WSL stands if Zet keeps the split format.
- **No hosted full-precision English export remains.** `distinctinteractive/laya-onnx` went offline on 2026-09-29. See F1 in decisions.md.
- **Pin every Hub artifact** by revision and SHA-256. Laya's own loaders support this (`laya/onnx_agent.py:72–78`).

---

## Temperatures

- **Load:** read `temperature` and `temperature_by_options` from `rl_agent_config.json` (`laya/onnx_agent.py:179–183`).
- **Bucket:** `"<type>:<size>"`, where size is one of `2`, `3-5`, `6-10`, `11+` (`laya/common.py:499–505`). A bucket temperature takes priority over the per-type one (`laya/onnx_agent.py:567`).
- **Clamp to [0.5, 5.0]** (`laya/common.py:508–520`). The code comment says a shipped `choice:11+` bucket was fitted at 0.1006, which would turn a 0.24 top probability into 0.99. Laya warns and clamps (`laya/onnx_agent.py:200–214`).
  - The MASSIVE 18-scenario question falls in that bucket, so its probabilities are known to be poorly calibrated even after clamping.
  - The conformal and Learn-then-Test guarantees don't depend on calibration, so they stay valid. Expect wide sets and a low automation rate.
- **Per-language temperatures are not in the checkpoint.** `lang_temperatures` is a constructor argument the caller supplies (`laya/onnx_agent.py:56`, `83–86`, `186–198`), and it applies only when `lang=` is passed (`laya/onnx_agent.py:568–571`).
  - By default Swedish and English get the same temperatures. Zet does not pass `lang_temperatures`, in line with Q3: use Laya's shipped temperatures, and let Zet's per-group calibration do the rest.

---

## Output format

This is Laya's decoded answer (`laya/onnx_agent.py:563–616`; the PyTorch path is identical at `laya/agent.py:869–926`; HTTP contract in `docs/http-api.md:79–110`):

| type | keys |
|---|---|
| choice | `choice` (argmax key), `probabilities` {option → p} |
| score | `score` (expected level Σ i·pᵢ, may fall between levels), `probabilities` {"0".."k-1" → p}, `legend` {"i" → level text} |
| noul | `noul` = p(true). **No `probabilities` field.** |
| all | `confidence`, `answer_confidence` = max(p), `action.act_probability` |

- **Everything is rounded to 4 decimals** with `round(..., 4)`, so a rounded `probabilities` dict need not sum to 1 within 1e-6.
  - Zet's ONNX backend computes its own float64 softmax.
  - The HTTP backend renormalises the rounded values and sets its contract tolerance to 5e-4 × k.
- **`confidence` is not a calibrated number.** For choice and score it is normalised entropy; for noul it is max(p). Laya's docs warn never to threshold it (`laya/common.py:470–497`, `docs/http-api.md:112–125`). Zet uses neither `confidence` nor `answer_confidence`; its selective threshold runs on its own top-option probability.
- **`act_probability`** comes from a separate act head (`laya/common.py:342–348`) and is unused in 0.1. Zet keeps it in `predictions.jsonl` as a possible 0.2 feature.

---

## Language detection and routing

- **`laya.lang.analyse`** (`laya/lang.py:585`) is dependency-free and torch-free (imports only `re`, `unicodedata`, `collections.abc`, `typing`). It returns `script`, `language`, `is_english`, `language_undecided`, `non_latin_fraction` and `mixed_segment`.
- **The detector is a router, not a language identifier.** Its stopword lists cover only 10 languages: `en, fr, de, es, pt, it, nl, ro, bn, az` (`laya/lang.py:56` onward). **Swedish is not identified.** (An earlier version of these notes said 9, leaving out `az`.)
- **Checked by running `laya/lang.py` from this commit:**

| input | `language` | `is_english` | `language_undecided` |
|---|---|---|---|
| "Kan ni skicka om förra månadens faktura? Inte bråttom." | None | False | True |
| "Jag vill inte ha pengarna tillbaka, bara en fungerande app." | None | **True** | True |
| "Hej, appen kraschar när jag loggar in" | None | False | True |
| "Tack!" | None | True | True |
| "Could you resend last month's invoice? No rush." | en | True | False |

- **Routing rule** (`laya/router.py:617–705`): non-Latin script goes to multilingual; Latin-script text that isn't English goes to multilingual; Latin-script text with an undecided language goes to `self.default`, which is **"english"** (`laya/router.py:370`, `697–704`).
  - So Swedish without å/ä/ö can be routed to the English checkpoint. That checkpoint collapses outside English (`laya/router.py:21–24`). This may explain the brief's "Swedish failing where English succeeded."
- **Consequences for Zet** (flagged in the summary):
  - Zet can't use `laya.lang.analyse` as its grouping key; Q6 needs a detector that names Swedish.
  - Zet shouldn't rely on Laya's routing for checkpoint choice either. A task pins its checkpoint (english / multilingual).
  - The documented escape hatch is `Router(default="multilingual")` or passing `lang=` (`laya/router.py:640–650`).

---

### Zet's extension (step 2)

`zet/_laya/lang.py` adds Swedish, Norwegian and Danish stopword lists. Words any other list already holds are left out, so they stay evidence for that language.

- **Laya's evidence rule needed a fix for Norwegian and Danish.** The rule only names a language on a word no other list holds. Norwegian and Danish share nearly all their function words, so neither was ever named, and "Jeg vil ikke have pengene tilbage, bare en app der virker." came back as English.
- **The family rule:** inside the Norwegian/Danish family, a word shared only by the family counts as evidence for it. When no unique word separates the two, the family code `da-no` is reported.
- **Checked against Laya's own tests:** all 94 of Laya's language assertions (`tests/test_lang_stats.py`, and the language section of `tests/test_router.py`) pass on the modified detector, the same as on the original.

## HTTP server

- **Endpoint:** `laya-serve` (`pyproject.toml:81`) serves `POST /v1/systemone` (`laya/serve.py:425`). The request is `{state, questions, model?}` and the response is `{model, answers, usage, routing}` (`docs/http-api.md:50–110`).
- **Pin the checkpoint:** a `model` value of `english` or `multilingual` fixes the checkpoint; any other value lets the router choose (`docs/http-api.md:71–77`). `LayaHttpBackend` always sends `model`.
- **No `lang` field in the request**, so per-language temperatures can't be selected over HTTP. That makes no difference for Zet (see Temperatures).
- **Mapping to Zet's shape:** probabilities are rounded (see above). For noul, `p(false) = 1 − noul`.
- **`examples/server.py`** is a different, larger demo server with `/predict` and `/predict/batch` (`examples/server.py:332`, `354`). Target `laya-serve` only.
- **Laya already serves a Jev-compatible `/v1/systemone`.** Zet 0.3's planned server of the same name must add Zet's `status/options/audit` fields, not duplicate Laya.

---

## Why the `laya` package can't be used directly

- **The package hard-depends on torch:** `dependencies = ["torch>=2.0.0", "transformers>=4.48.0", …]` (`pyproject.toml:27–33`).
- **`laya/common.py` imports torch at module top** (`laya/common.py:12–15`), and `onnx_agent.py` imports from it (`laya/onnx_agent.py:15–29`).
- **The ONNX path still uses torch:** `collate_items` builds torch tensors that are then `.numpy()`'d (`laya/common.py:527–569`, `laya/onnx_agent.py:662`). It also imports `Agent` from `laya/agent.py` for validation (`laya/onnx_agent.py:629`).
- **It needs `transformers`** for the tokenizer (`laya/onnx_agent.py:97`).

So, as decided in Q1, Zet copies the torch-free logic and sends an upstream PR splitting `common.py`.

### What Zet copies, into `zet/_laya/`

| Zet file | from | lines | change |
|---|---|---|---|
| `sequence.py` | `laya/common.py` | 17–19, 74–223, 470–520 | tokenizer calls via `tokenizers`; question-token cache dropped |
| `collate.py` | `laya/common.py` | 527–569 | numpy, no torch |
| `questions.py` | `laya/agent.py` 618–~700, `laya/onnx_agent.py` 217–231 | | standalone functions |
| `decode.py` | `laya/onnx_agent.py` | 563–616 | returns unrounded float64 probabilities |
| `lang.py` | `laya/lang.py` | whole file | kept for routing diagnostics only (see flags) |

### Licensing facts

- **License:** Laya is Apache-2.0 (`LICENSE`).
- **No NOTICE file, and no per-file copyright headers** in `laya/*.py`. Hub model cards credit "Convai Innovations" and "Copyright ConvAI Innovations".

---

## `embed()` (decided in Q2)

- **Definition:** run the encoder on the state alone (`[CLS] state [SEP]`, no question) and mean-pool `last_hidden_state` over `attention_mask`.
- **Caveat:** the model never sees a question-less state in training or inference (every row includes the question, see Input format). The embedding is therefore a generic encoder embedding, not the representation the head decides on. That suits nearest-neighbour memory in 0.2, but don't read it as "what Laya thought."
- **Requires the split format**, or a single graph that also outputs the encoder hidden states. With single-graph weights (the current multilingual default), `embed()` returns `None`, which the 0.1 interface allows.

---

## Differences from the brief

1. **"One forward pass" means one batched call:** N questions are N encoder rows (Input format).
2. **The pre-converted split weights are English-only and 8-bit.** A full-precision multilingual export in split format doesn't exist yet. A full-precision single-graph one does (ONNX formats).
3. **Probabilities are rounded to 4 decimals** in every Laya output. Zet computes its own softmax, and the HTTP backend needs a looser tolerance than 1e-6 (Output format).
4. **noul returns only p(true)**; Zet builds `{"false": 1−p, "true": p}` (Output format).
5. **Laya's language detector doesn't identify Swedish**, and can route Swedish to the English checkpoint. Norwegian and Danish are affected the same way (Language detection).
6. **`laya[onnx]` doesn't give a torch-free install**; the base package requires torch (Why the package can't be used directly).
7. **Laya already serves `POST /v1/systemone`** (HTTP server).
8. **Per-language temperatures aren't shipped** with any checkpoint (Temperatures).
9. **There are no copyright headers to keep in copied files.** The brief's rule 4 needs a header wording for that case (Licensing facts).
