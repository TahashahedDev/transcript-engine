# Transcript Engine — How It Works

*A practical guide to the architecture, speech recognition, speaker diarization, processing pipeline, and engineering decisions.*

*Revision note: this version expands the sections flagged in manager review (audio preparation, diarization mechanics, VAD, Whisper model sizing, wav2vec2, TDT, Conformer, speaker embeddings, and "why our own diarization") without changing the underlying architecture described in the original guide.*

---

## 1. The 30-Second Explanation

The Transcript Engine takes an audio recording and converts speech into a timestamped transcript. It identifies different speakers using speaker diarization, and generates structured meeting artifacts (summary, action items, decisions, questions, timeline).

```
Audio
  ↓
Audio preparation (convert to 16kHz mono WAV)   — Section 3
  ↓
Chunking (split long audio into bounded pieces) — Section 6
  ↓
Speech-to-text (ASR)          ← runs in parallel with ↓
Speaker diarization  ─────────┘                  — Sections 7–8
  ↓
Match timestamps → speaker-labeled transcript    — Section 7
  ↓
Meeting intelligence (summary, action items, decisions, questions, timeline)
```

---

## 2. What Actually Happens to Speech?

```
Audio waveform → acoustic patterns → speech recognition model → tokens/text → timestamps → transcript
```

An audio waveform is just pressure-over-time. A neural acoustic model converts short windows of that waveform into a sequence of learned tokens (roughly: sub-word pieces), which decode into text. Modern ASR models also emit a timestamp per token/word as a side effect of decoding — so you get *what was said* and *when* in a single pass.

**The critical distinction:** ASR and speaker identification are two separate problems, solved by two separate models. ASR never asks "whose voice is this?" — that's diarization's job (Sections 8–9).

---

## 3. Audio Preparation: What "Processed Audio" Actually Means

**What happens:** every uploaded file (any audio or video container ffmpeg can read) is converted, before any model sees it, into **16 kHz, mono, 16-bit PCM WAV** — done by `AudioPreprocessor.prepare()` (`transcript_engine/audio/preprocessor.py`) via a single `ffmpeg` call:

```
ffmpeg -i <input> -ar 16000 -ac 1 -c:a pcm_s16le -vn <output>.wav
```

| Flag | Effect |
|---|---|
| `-ar 16000` | Resample to 16,000 Hz |
| `-ac 1` | Mix down to mono (1 channel) |
| `-c:a pcm_s16le` | Uncompressed 16-bit PCM (no lossy codec artifacts going into the model) |
| `-vn` | Drop any video stream — only audio is kept |

**"Unprocessed" vs. "processed" audio, concretely:**

| | Unprocessed (upload) | Processed (`*_prepared.wav`) |
|---|---|---|
| Format | Anything ffmpeg supports: mp3, mp4, m4a, wav, aac, etc. | WAV (PCM) |
| Sample rate | Whatever the source used (often 44.1 kHz or 48 kHz) | 16,000 Hz |
| Channels | Often stereo (2ch) or more | Mono (1ch) |
| Bit depth/encoding | Often lossy-compressed | 16-bit PCM, uncompressed |

**Why 16 kHz mono specifically — not a stylistic choice, it's a model-compatibility requirement:**
- Both Parakeet and Whisper were **trained on 16 kHz mono audio**. Feeding them anything else means the model either implicitly resamples (extra latency, occasional quality loss) or performs worse — the model's learned filters expect a specific frequency range and sample density.
- Human speech's useful frequency content mostly sits below 8 kHz. By the Nyquist theorem, a 16 kHz sample rate is sufficient to represent everything up to 8 kHz — so no speech-relevant information is lost by downsampling from a 44.1/48 kHz source.
- Mono removes a needless second channel: speech content is typically duplicated (or near-duplicated) across stereo channels anyway, so keeping stereo would just double the data the model has to process for no accuracy gain.
- Uncompressed PCM avoids re-introducing lossy-codec compression artifacts (e.g. mp3 quantization noise) right before the model runs inference.

**What this project does *not* do:** there is no loudness normalization, noise reduction, or silence trimming in `AudioPreprocessor` — it is strictly a format/sample-rate/channel conversion step. Any "cleanup" of the audio happens implicitly inside the downstream models (see the VAD note below), not as an explicit preprocessing stage in this codebase.

**Where VAD (Voice Activity Detection) actually happens in this project:**

VAD is *not* a step in `AudioPreprocessor` — it happens inside two different downstream components, for two different purposes:

| Component | Uses VAD? | Purpose |
|---|---|---|
| `AudioPreprocessor` (Section 3) | No | Pure format conversion only |
| WhisperX/CTranslate2 path (`vad_filter=True`, `whisperx_engine.py`) | Yes | Skips silent regions before transcription, so the model doesn't waste compute decoding silence and doesn't hallucinate text over dead air |
| pyannote diarization pipeline (Section 8) | Yes, internally | The pipeline's segmentation model implicitly separates speech from non-speech as the first step of clustering voices — this is not a separate call our code makes, it's built into `pyannote/speaker-diarization-3.1` |
| Parakeet path (`parakeet_engine.py`) | No explicit VAD filter | Chunk audio is sent to the model as-is; Parakeet's own acoustic model absorbs silence handling |

Further reading: [Voice Activity Detection (Wikipedia)](https://en.wikipedia.org/wiki/Voice_activity_detection) · [ffmpeg documentation](https://ffmpeg.org/ffmpeg.html)

---

## 4. What Is ASR?

**ASR = Automatic Speech Recognition.** It's called "automatic" to distinguish it from earlier approaches to transcription that required a human transcriber or hand-built rule/template systems — ASR means a trained model performs the speech-to-text conversion end-to-end without a person in the loop or hand-written grammar rules.

| ASR answers | ASR does NOT answer |
|---|---|
| "What was said?" | "Who said it?" |
| speech → text | speaker → identity |

ASR outputs plain text with timestamps, with no notion of speaker turns. Speaker labeling is bolted on afterward by a completely different system (diarization).

---

## 5. Whisper / WhisperX

**Whisper** (OpenAI) is a general-purpose speech recognition model trained on ~680k hours of multilingual audio. It became popular because it's accurate out-of-the-box across many languages and accents without fine-tuning.

**Whisper model sizes** (publicly documented by OpenAI — general knowledge, not a measurement from this project). Approximate parameter counts and typical full-precision GPU memory footprints:

| Model | Parameters | Approx. VRAM (fp16) | Relative speed |
|---|---|---|---|
| tiny | 39M | ~1 GB | fastest, least accurate |
| base | 74M | ~1 GB | fast |
| small | 244M | ~2 GB | balanced |
| medium | 769M | ~5 GB | slower, more accurate |
| large-v2 / large-v3 | 1550M | ~10 GB | slowest, most accurate |

These are approximate, widely-cited figures for the reference implementation — **not benchmarked in this project**, and not directly comparable to this project's numbers because of the quantization point below.

**What this project actually runs:** `transcript_engine/config/settings.py` configures **`large-v3-turbo`** as the default WhisperX model, with quality-mode variants selecting `Systran/faster-distil-whisper-large-v3` (Fast mode) or full `large-v3` (High Accuracy / Archive modes). Critically, this project runs WhisperX through **CTranslate2 with int8 quantization on CPU** by default (not fp16 on GPU) — int8 quantization roughly quarters the memory footprint versus fp16 at some accuracy cost, which is why the CPU path is viable at all without a dedicated Whisper GPU budget. An MLX (Apple Silicon GPU) backend exists behind `TE_USE_MLX=1` but has a documented thermal-throttling issue on 8GB M1 hardware, so CT2/CPU is the safe default.

**WhisperX** is a wrapper around Whisper that adds a separate **forced-alignment** step to produce accurate word-level timestamps — Whisper alone only gives segment-level timestamps.

**Segment-level vs. word-level timestamps, concretely:**

```
Segment-level (raw Whisper output):
  [00:00 – 00:04.2] "we should launch this next week"
  → one timestamp range for the whole sentence; no per-word timing

Word-level (after WhisperX forced alignment):
  we (00:00.0–00:00.3)  should (00:00.3–00:00.7)  launch (00:00.7–00:01.1)  this (00:01.1–00:01.3)
  next (00:01.3–00:01.6)  week (00:01.6–00:02.0)
  → every word has its own start/end time
```

Word-level timestamps are what make speaker-attribution possible at word granularity (Section 7) — you can't match a whole 4-second sentence to a speaker segment if two different people spoke within it, but you can match individual words.

**What is the wav2vec2 alignment model doing?** wav2vec2 (Meta AI) is a separate, smaller neural network trained for **forced alignment**: given text that's already known to be correct (Whisper's output) and the audio it came from, it predicts precisely where in the audio each phoneme/word boundary falls. It doesn't do speech recognition itself — it's only answering "given this text is right, when exactly was each word spoken?" This is why WhisperX is a two-model pipeline (Whisper for text, wav2vec2 for timing) rather than a single model.

**In this codebase:** `transcript_engine/transcription/whisperx_engine.py` is the legacy/fallback ASR path, selected when `TE_ASR_BACKEND != parakeet` (default is `"whisper"` at the config level, but Parakeet is the actively used production path per the pipeline_config). It runs on CTranslate2 (CPU, int8) by default; an MLX (Apple Silicon GPU) backend is available behind `TE_USE_MLX=1` but is documented in-code as having a thermal-throttling bug on 8GB M1 hardware, which is why CT2 is the safe default.

| Characteristic | WhisperX in this project |
|---|---|
| Word timestamps | Requires a separate wav2vec2 alignment pass |
| Diarization | Not built-in here — diarization is a fully separate pyannote step |
| GPU requirement | Runs on CPU (CT2) or Apple Metal (MLX, opt-in) |
| Role today | Legacy/fallback backend |

Whisper/WhisperX does **not** perform speaker diarization by itself in this system — that would require WhisperX's own diarization integration, which this project deliberately does not use (see Section 8a: version-pinning conflicts made a manually-wired pyannote pipeline the safer choice).

Further reading: [Whisper paper (OpenAI)](https://cdn.openai.com/papers/whisper.pdf) · [wav2vec 2.0 paper (Meta AI)](https://arxiv.org/abs/2006.11477) · [WhisperX paper](https://arxiv.org/abs/2303.00747)

---

## 6. NVIDIA Parakeet / NeMo

**NVIDIA NeMo** is the toolkit/framework used to load and run NVIDIA's speech models. **Parakeet** is the specific ASR model running inside NeMo.

**Model actually configured:** `nvidia/parakeet-tdt-0.6b-v2` (default; overridable via `TE_PARAKEET_MODEL`).

This is the **active production ASR path** in the system.

| Parakeet | NeMo |
|---|---|
| The speech recognition model | The framework that loads and runs it |
| Answers "what was said, and when" | Provides `EncDecRNNTBPEModel.from_pretrained()` + `.transcribe()` |

**Why Parakeet:**
- Its TDT (Token-and-Duration Transducer) decoder produces **word-level timestamps natively** — no separate alignment model needed, unlike Whisper/WhisperX.
- Full-attention Conformer encoder, tuned for GPU throughput.
- Runs on CUDA GPUs, which is the deployment target for this project (Section 16).

**What is TDT (Token-and-Duration Transducer)?**

TDT is a decoder architecture — a variant of the standard RNN-Transducer (RNN-T) used in streaming/online ASR. The key difference:

- A standard RNN-T decoder predicts **one token per audio frame** (or a blank), stepping forward one frame at a time — this couples decoding speed directly to audio length and produces timestamps only indirectly (from which frame each token was emitted on).
- TDT predicts **a token *and* a duration** at each decoding step — the duration tells the decoder how many frames to skip before the next prediction. This means:
  1. Decoding is faster, because the model can skip multiple frames per step instead of stepping through every single one (fewer decoder calls for the same audio).
  2. **Word-level timing falls out of the decoding process itself** — the duration prediction *is* timing information, so there's no need for a second alignment model (like wav2vec2) to figure out word boundaries after the fact.

This is the direct answer to "why doesn't Parakeet need forced alignment the way WhisperX does": the timestamp isn't a post-hoc alignment problem, it's a first-class output of the decoder.

**What is a Conformer, conceptually?**

Conformer ("Convolution-augmented Transformer") is the encoder architecture that turns the audio waveform into the internal representation the decoder reads from. It combines two mechanisms that are individually good at different things:

- **Self-attention** (from Transformers) — good at capturing long-range context: e.g., relating a sound at the start of a sentence to one near the end, or using broader context to disambiguate an acoustically ambiguous sound.
- **Convolution** — good at capturing local acoustic patterns: the fine-grained, short-timescale structure of phonemes (a few milliseconds to tens of milliseconds), which is naturally local and where convolutions are efficient and translation-invariant.

By stacking convolution and self-attention in the same encoder block, Conformer captures both the local acoustic detail and the long-range context that speech recognition needs, which is why it became the standard encoder for high-accuracy production ASR models (used by Parakeet, and also by later Whisper-family and other industrial ASR systems).

**Limitation:** like most ASR models, a single pass has a practical audio-length ceiling before memory and latency become impractical — this is the direct motivation for chunking (Section 7).

Further reading: [Conformer paper](https://arxiv.org/abs/2005.08100) · [RNN-Transducer overview](https://arxiv.org/abs/1211.3711) · [NVIDIA NeMo docs](https://docs.nvidia.com/nemo-framework/)

---

## 7. Why We Use Chunking

**Question:** Why not send a 1-hour recording directly into the model?

Longer audio → longer sequence → more GPU memory and compute per pass. Unbounded audio length means unbounded (and unpredictable) memory usage. So the system splits long recordings into bounded chunks sized to the available VRAM.

**Actual chunk-size table** (`transcript_engine/gpu/hardware.py`), selected automatically from detected free VRAM:

| Free VRAM | Chunk length |
|---|---|
| < 4 GB | 90s |
| 4–6 GB | 180s |
| 6–8 GB | 300s |
| 8–12 GB | 480s |
| 12–16 GB | 720s |
| 16–24 GB | 900s |
| 24 GB+ | 1320s |

```
1-hour recording
  ↓
split into N VRAM-sized chunks
  ↓
transcribe each chunk independently
  ↓
stitch results back together
```

**Why overlap:** a word can land exactly on a chunk boundary.

```
Without overlap:
  chunk 1: "...we will dis"
  chunk 2: "cuss the..."

With overlap:
  chunk 1 (+0.5s padding): "...we will discuss the"
  chunk 2 (+0.5s padding): "we will discuss the..."
```

The system extends each chunk by **0.5 seconds** (`DEFAULT_OVERLAP_SECONDS`) of extra audio on both sides, so a boundary word is fully audible in at least one chunk.

**Removing duplicates:** each chunk has a *nominal* window (its real, non-overlapping slice of the timeline) and a slightly larger *padded* window (what was actually sent to the model). A word is kept only if its start time falls inside the chunk's **nominal** window:

```
keep word if:  nominal_start <= word.start < nominal_end
```

This guarantees every word is attributed to exactly one chunk — no duplicates, no gaps.

**Short audio uses the same path as long audio.** `split_wav_with_overlap()` degrades to a single whole-file "chunk" when the recording is shorter than the chunk size — it does not copy or split the file, but it does route through the same OOM-retry logic (Section 14) as a genuinely multi-chunk job. There is no separate, unprotected fast path for short recordings.

---

## 8. The Most Important Interview Question: "How Do You Know Who Spoke?"

**ASR answers:** "What was said?"
**Diarization answers:** "Who was speaking when it was said?"

```
Audio
  ↓
ASR → words + timestamps
  ↓
Diarization → speaker segments + timestamps
  ↓
Match timestamps (max time-overlap)
  ↓
Assign speaker label to each word
  ↓
Speaker 1: "We should launch this next week."
Speaker 2: "I agree. Let's finish testing first."
```

**The system does not know that a voice belongs to "John" or "Sarah."** It only clusters voices into anonymous identities like `SPEAKER_00`, `SPEAKER_01`. These raw labels are remapped to display names `Speaker 1`, `Speaker 2`, etc. — there is no real identity recognition (Section 10).

**Matching algorithm** (`transcript_engine/merger/merger.py`, `TranscriptMerger._assign_speakers()`): for each ASR word, find the diarization segment with **maximum time overlap**. If no diarization segment overlaps that word at all, fall back to the last known speaker. This is an O(n+m) sliding-pointer scan across both timestamp sequences, not a nearest-neighbor or midpoint heuristic.

**Why max-overlap instead of a simpler heuristic?** A word has a *duration*, not just a start point — "nearest timestamp" or "nearest midpoint" approaches compare a single point to segment boundaries and can pick the wrong speaker when a word straddles a speaker change (e.g., someone starts talking right as another person's word is still finishing). Max-overlap instead asks "which speaker was talking for the largest fraction of this word's actual duration?" — which is the more accurate question when speaker turns and word boundaries don't line up perfectly.

---

## 9. What Is Speaker Diarization?

**Definition:** speaker diarization determines *who spoke when* — without knowing *who* anyone actually is.

Conceptually, the pipeline stage-by-stage:

1. **Voice activity detection** — separate speech from silence/noise (this is the VAD step referenced in Section 3; inside `pyannote/speaker-diarization-3.1` it's implicit in the pipeline's segmentation model, not a separate call this project makes).
2. **Speaker embedding extraction** — for each detected speech region, run a neural network that outputs a fixed-length numeric vector (an "embedding") summarizing the acoustic characteristics of that voice — pitch, timbre, resonance, and other speaker-discriminative features, compressed into a point in a high-dimensional space.
3. **Clustering** — group embeddings that are close together in that space. Two regions spoken by the same person should produce embeddings that are close (by cosine similarity or a similar distance metric); different speakers should produce embeddings that are far apart.
4. **Segment assignment** — each cluster becomes one anonymous "speaker" (`SPEAKER_00`, `SPEAKER_01`, ...).
5. **Emit timestamped segments** — the final output is a list of `(start_time, end_time, speaker_id)` tuples.

```
00:00–00:08 → Speaker 1
00:08–00:15 → Speaker 2
00:15–00:22 → Speaker 1
```

**On speaker embeddings and their dimensionality — stated honestly, not invented:** the embedding model and its exact vector dimension are **internal to the pretrained `pyannote/speaker-diarization-3.1` pipeline** — this project calls `Pipeline.from_pretrained(config.model_id)` (`transcript_engine/model_registry/registry.py`) and does not configure, load, or reference a specific embedding model or dimension anywhere in the codebase. Do not state a specific number (e.g. "256-dim" or "512-dim") as fact for this project unless it has actually been checked against the resolved pipeline's config at the specific pinned version in use — pyannote's public model cards document their embedding component, and that's the authoritative source to check, not this document. What *is* safe to say in an interview: it's a fixed-length real-valued vector, produced by a neural network trained specifically to make same-speaker vectors close together and different-speaker vectors far apart in that space — the standard "speaker embedding" concept used across the field (this general family of models is often described as computing an "x-vector"-style or similar speaker representation).

**In this project:** diarization uses `pyannote.audio`, model `pyannote/speaker-diarization-3.1`, run through a manually-wired pipeline (`transcript_engine/diarization/pyannote_engine.py`) rather than WhisperX's built-in diarization, specifically to avoid a dependency version conflict (expanded in Section 9a below).

**Performance trick:** diarization runs on **CPU in a background thread** while ASR runs on the **GPU**, in parallel (`ThreadPoolExecutor`, orchestrator + `parallel_diarizer.py`), so the two don't compete for VRAM or serialize unnecessarily. On a CUDA machine, pyannote can also run on GPU after Parakeet finishes (code comments note pyannote CPU takes ~2–5 min vs. ~15–30s on GPU for a 90-minute recording — a 15–20x difference — though this, like Section 12's numbers, should be treated as a design rationale noted in code comments rather than a captured, reproducible benchmark).

Those speaker segments are then matched against ASR word timestamps using the max-overlap algorithm from Section 8.

Further reading: [pyannote.audio](https://github.com/pyannote/pyannote-audio) · [Speaker diarization overview (survey paper)](https://arxiv.org/abs/2101.09624) · [x-vector speaker embeddings](https://danielpovey.com/files/2018_icassp_xvectors.pdf)

### 9a. Why Our Own Diarization Instead of WhisperX's Built-In Diarization?

WhisperX ships an optional built-in diarization integration (it also wraps pyannote internally). This project does **not** use that integration, and instead calls pyannote directly through its own pipeline module. The reasons, honestly separated into what's actually verified vs. what's just architectural preference:

**Confirmed engineering reason (dependency conflict):** WhisperX pins specific versions of `pyannote.audio` and related dependencies (torch, torchaudio, huggingface_hub) internally. Running the exact pyannote model/version this project needs alongside those pins produced dependency version conflicts. Wiring pyannote manually removes that coupling — this project can pin pyannote's own version independently of whatever WhisperX's package requires.

**Architectural reasons, not just a version workaround:**
- **Independent replaceability.** Because ASR (Parakeet or WhisperX) and diarization (pyannote) are wired as two decoupled components joined only by the merger (Section 8), either one can be swapped, upgraded, or removed independently. If diarization were baked into the ASR wrapper, replacing the ASR backend (which this project already did — WhisperX → Parakeet as the default) would also mean losing or re-plumbing diarization.
- **Parallel execution.** A manually-wired pyannote call is straightforward to run on its own thread, in parallel with ASR (Section 9's "performance trick"). A diarization step baked into the ASR library's own call path is harder to pull out and run concurrently.

**Is "our own diarization" more *accurate* than WhisperX's built-in diarization?** No claim is made here that it is — both integrations would be calling into pyannote's models one way or another, so there is no evidence in this project that accuracy differs. **The honest engineering answer for an interview is:** "we chose a decoupled, manually-wired pyannote pipeline for dependency isolation and independent component swapping — not because we measured it as more accurate than WhisperX's own diarization integration."

---

## 10. Diarization vs. Speaker Identification

This distinction is explicit and important — do not conflate them.

| | Diarization (implemented) | Speaker identification (NOT implemented) |
|---|---|---|
| Claim | "Speaker A spoke here." | "Speaker A is John." |
| Requires | Voice clustering | A pre-enrolled voiceprint database + matching |
| Output | `SPEAKER_00` → displayed as `Speaker 1` | A real name |

The system has no voiceprint enrollment, no name matching, and no persistent speaker identity across recordings. Every job's speaker labels are anonymous and local to that job only.

**What would be required to add speaker identification (future, not implemented):**

1. **Enrollment** — a user records or uploads a short reference voice sample for each person they want recognized.
2. **Reference embedding** — run that sample through the same (or a compatible) speaker-embedding model used in diarization, producing a reference vector per enrolled person.
3. **Stored voice profile** — persist that reference embedding (e.g., in the existing-but-unwired PostgreSQL schema, Section 16) associated with a name.
4. **Similarity matching** — for each diarized cluster in a new recording, compute similarity (e.g. cosine similarity) between that cluster's embedding and every stored reference embedding.
5. **Confidence threshold** — only accept a name match above a chosen similarity threshold, to avoid confidently mislabeling a stranger as an enrolled person.
6. **Unknown-speaker fallback** — anyone below the threshold stays an anonymous `Speaker N`, rather than being forced into the nearest enrolled name.

None of this exists in the current codebase — it's listed here purely as the standard shape such a feature would take, for interview "what would you build next" questions.

---

## 11. End-to-End Example

Two people have a 1-hour meeting.

```
Audio (60 min)
  ↓
Chunked into ~5–8 pieces (VRAM-dependent) with 0.5s overlap
  ↓
Parakeet transcribes each chunk → words + timestamps
  ↓
pyannote diarizes the full audio (in parallel, on CPU) → speaker segments
  ↓
TranscriptMerger matches word timestamps to speaker segments (max-overlap)
  ↓
Speaker-labeled transcript
```

Result:

```
Speaker 1:
"We should launch this next week."

Speaker 2:
"I agree. Let's finish testing first."
```

From here, the meeting intelligence engine (Section 15) runs on top of this speaker-labeled transcript to extract structured artifacts.

---

## 12. Why This Architecture?

| Problem | Decision | Why |
|---|---|---|
| Long audio exceeds GPU memory | VRAM-sized chunking | Bounded, predictable memory use |
| Words split across chunk boundaries | 0.5s overlap + nominal-window ownership | No lost or duplicated words |
| Speech → text | Parakeet (NeMo) | Native word timestamps, GPU-optimized, current production ASR |
| "Who spoke?" is a different problem than "what was said?" | Separate pyannote diarization pipeline | Keeps ASR and speaker modeling independently swappable |
| ASR and diarization are both slow | Run diarization (CPU) parallel to ASR (GPU) | Avoids serializing two independent, resource-disjoint jobs |
| Long GPU jobs risk OOM | Bounded chunk retry with automatic size halving | Recovers from a single chunk's OOM without failing the whole job |
| Large recordings shouldn't block the API | Background job + SSE progress streaming | Keeps the HTTP API responsive during long jobs |

---

## 13. Performance

**No validated real-world Parakeet+GPU benchmark exists yet in this project.** `DEPLOYMENT.md` documents a **target**, not a measured result: roughly 2–3 minutes for a 90-minute recording on an RTX 4090-class GPU. The engine logs real-time-factor (RTF) at runtime, but no benchmark run has been captured and recorded.

*(Older measured numbers exist for a previous WhisperX/MLX pipeline on Apple Silicon — ~19–20 min for a 60-min recording on an 8GB M1 — but that was a different backend on different hardware and should not be quoted as current Parakeet performance.)*

What determines performance in the current system:
- GPU model and available VRAM (drives chunk size — bigger chunks, fewer round trips)
- ASR backend (Parakeet vs. WhisperX)
- Diarization running in parallel vs. serially
- Total audio length
- CPU throughput for diarization (runs concurrently, but still a real cost)

**Honest status: real-world validation on production GPU hardware is the next required step**, not a completed benchmark.

Further reading: [Real-time factor (RTF) definition](https://en.wikipedia.org/wiki/Real-time_computing#Real-time_factor)

---

## 14. Reliability / OOM Protection

**Problem:** a long recording, or a large chunk, can exceed GPU memory mid-job. Less obviously: **two GPU-heavy stages of the same job can also exceed memory by running at once**, even when each is individually sized to fit.

**The concurrency problem, specifically:** Parakeet transcription and pyannote diarization can both run on CUDA, and the orchestrator (`transcript_engine/pipeline/orchestrator.py`) launches them as concurrent threads by design, for wall-clock speed. On a large-VRAM GPU that headroom absorbs both at once. On a smaller card (the 8 GB RTX 3060 Ti target, specifically) it doesn't — Parakeet's chunk size is computed from free VRAM *before* diarization's own model and activations land on the same device, so the two engines' independently-safe budgets can collide in practice. This was a real production OOM source, not a hypothetical.

**Fix:** a single process-wide `GPU_COMPUTE_LOCK` (`transcript_engine/gpu/hardware.py`), held by both `ParakeetEngine` and `PyannoteEngine` whenever pyannote is on CUDA. Whichever engine grabs it runs its GPU work to completion before the other starts. The orchestrator still launches both threads immediately (so CPU-side setup pipelines), but actual GPU compute never overlaps. Diarization on CPU (small/no GPU) is unaffected — the lock is a no-op there and true parallelism is preserved.

**Solution, layered:**

| Problem | Solution |
|---|---|
| Unbounded memory from long audio | VRAM-sized chunk bounds (Section 7) |
| Two GPU engines allocating at once | `GPU_COMPUTE_LOCK` serializes Parakeet transcription against GPU-placed diarization |
| Stale VRAM reading | Free VRAM is re-detected at the start of every job, not cached from the first model load |
| Memory fragmentation across jobs | `PYTORCH_CUDA_ALLOC_CONF` tuning + `gc.collect()` / `torch.cuda.empty_cache()` between chunks |
| A single chunk OOMs (any audio length — see Section 7) | Automatic retry: halve that chunk's audio and retry, up to a max retry count |
| Diarization itself OOMs on CUDA | One-shot fallback: move the pipeline to CPU and retry once, rather than failing the job |
| Leftover temp files on failure | Explicit cleanup on every exit path (success, OOM, and unexpected exceptions) |
| Stale output/temp directories | Hourly background cleanup task with a TTL |
| Slow GPU matrix ops | TF32 enabled on Ampere+ GPUs; cuDNN benchmark mode enabled |

**Hardware-agnostic by design:** none of the above is tuned to one specific GPU. Chunk size and the diarization CPU/GPU placement decision are both derived from VRAM detected at runtime (`detect_gpu()`), not a hardcoded card. The same code should behave safely on a 3060 Ti, a 4090, or an L40S — it just picks different chunk sizes and, if a diarization OOM does slip through, falls back to CPU regardless of which card it's on.

**Stuck-job watchdog (implemented, with a caveat).** A watchdog thread per job fails the job if the pipeline emits *no* progress for a configurable window (default 45 min, `TE_API_STALL_TIMEOUT_MINUTES`). It deliberately watches only the real pipeline progress callback — the synthetic progress ticker and resource monitor run in their own threads and would keep a heartbeat alive even while the pipeline is wedged, so a heartbeat that survives the hang cannot detect the hang.

**The caveat, stated plainly:** Python cannot safely kill a running thread. The watchdog moves the *job* to a terminal `failed` state so the API and UI stop waiting forever, but the wedged worker thread still holds the single-worker executor, and later jobs stay queued until the process restarts. That limitation is logged explicitly as an operator signal. Recovering executor throughput automatically would require moving pipeline execution into a subprocess — a real architectural change, not done here.

---

## 15. Outputs

Generated by `transcript_engine/intelligence/engine.py` (`MeetingIntelligenceEngine`), which runs rule-based extractors over the speaker-labeled transcript:

| Artifact | File |
|---|---|
| Transcript (speaker-labeled) | `.transcript.*` |
| Summary | `.summary.md` |
| Action items | `.action_items.md` |
| Decisions | `.decisions.md` |
| Questions | `.questions.md` |
| Timeline | `.timeline.md` |

An optional AI-enhanced mode exists (`ai_enhanced` flag) for a language-model pass on top of the rule-based extraction, when configured. Separately, a smaller `TranscriptIntelligenceProcessor` component handles only sentence/punctuation cleanup and explicitly never rewrites, summarizes, or invents content — it is not the same system as the meeting-intelligence artifacts above.

---

## 16. Frontend

```
Upload → configure (profile, timestamp mode, quality mode) → submit → watch live progress (SSE) → read transcript → download artifacts
```

The user uploads a recording, picks a profile (domain vocabulary — e.g. generic vs. banking), a timestamp display mode (none / speaker / paragraph / minute), and a quality mode. Available quality-mode options differ by active ASR backend: Parakeet exposes **Quick** and **Standard** (Standard adds an AI grammar pass on top of Quick); Whisper exposes **Fast / Balanced / High Accuracy / Archive**. Both Parakeet modes run the *same* ASR model — the mode only changes which post-processors run afterward, never the transcription model itself (`PIPELINE_MODES` in `transcript_engine/config/settings.py`). Progress streams live via Server-Sent Events, and on completion the transcript and generated artifacts are available for download.

---

## 17. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Next.js (React, TypeScript) | Upload, progress UI, transcript viewer |
| Backend API | FastAPI | Job orchestration, SSE progress, REST endpoints |
| ASR (production) | NVIDIA Parakeet TDT (via NeMo) | Speech-to-text with native word timestamps |
| ASR (legacy/fallback) | Whisper / WhisperX (CTranslate2 or MLX) | Alternate speech-to-text backend |
| Diarization | pyannote.audio | Speaker segmentation |
| GPU management | Custom hardware module (`transcript_engine/gpu/hardware.py`) | Per-job VRAM detection, TF32/cuDNN tuning, chunk sizing, cross-engine GPU lock |
| Database | PostgreSQL + Alembic (schema exists) | **Not currently wired to the live job store** — jobs are in-memory today |
| Deployment | Plain Ubuntu GPU server, `start_server.sh` | No Docker, no systemd — foreground or `nohup`/`tmux` |

Further reading: [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---

## 18. What Can Go Wrong?

| Problem | What happens | Protection |
|---|---|---|
| GPU runs out of memory mid-chunk | Chunk transcription fails | Automatic retry with the chunk halved in size |
| Parakeet and GPU diarization allocate at the same time | Would otherwise OOM on smaller cards | `GPU_COMPUTE_LOCK` serializes the two engines' GPU work (Section 14) |
| Diarization itself OOMs on CUDA | Diarization call fails | One-shot retry on CPU before failing the job |
| A non-OOM error hits mid-chunk | Exception propagates | Cleanup runs on every exit path so temp files don't leak |
| Corrupt or unreadable audio upload | Upload/preprocessing fails | Validated at upload; errors clean up partial files and job records |
| Very long recording | Memory pressure across the whole pipeline | Bounded chunking + parallel diarization keep memory predictable |
| Diarization fails entirely | No speaker segments available | Falls back to a single `SPEAKER_00` segment spanning the full audio |
| Frontend disconnects mid-job | Client stops receiving progress | Job continues server-side; reconnecting resumes polling the same job ID |
| Server restarts mid-job | In-memory job store is lost | **Not currently protected** — no persistence across restarts today; the UI shows "this job no longer exists" rather than hanging |
| A fresh install resolves different library versions | Diarization breaks in ways that only appear on the new machine | `whisperx` and `pyannote.audio` are pinned as a compatible pair, and the pyannote API differences are handled in one place (below) |

### The pyannote version trap

Worth calling out, because it is the failure mode most likely to bite a fresh
GPU box rather than the machine the code was written on.

Two things this project touches moved between pyannote 3.x and 4.x:

| | pyannote 3.x | pyannote 4.x |
|---|---|---|
| Token keyword on `Pipeline.from_pretrained` | `use_auth_token` | `token` |
| Audio decoding backend | torchaudio / soundfile | torchcodec |

Passing the wrong token keyword raises `TypeError: unexpected keyword argument`
— at job time, not install time. And because the message contains the word
"token", it is easy to misread as a credentials problem and go fix a setting
that was already correct.

Two defences, both in `transcript_engine/diarization/compat.py`:

1. `load_pretrained_pipeline` inspects the actual function signature and passes
   whichever keyword that build accepts, so both majors work.
2. `apply_segmentation_step` guards the private `_segmentation` handle used to
   tune the segmentation step. If a release restructures those internals, it
   falls back to pyannote's default step — slower, still correct — instead of
   raising mid-job.

The version pin and the runtime shims are deliberately belt-and-braces: the pin
makes a fresh install reproducible, the shims keep a future bump from being a
hard failure.

---

## 19. Limitations

Stated plainly, not softened:

- **Diarization is not perfect.** Overlapping speech, similar-sounding voices, and short interjections can be misattributed or missed.
- **`Speaker 1` is not a name.** The system has no concept of real identity — only anonymous voice clusters.
- **Overlapping speech is genuinely hard** for both ASR and diarization; simultaneous talkers degrade both transcription and speaker attribution.
- **Audio quality matters a lot.** Background noise, low bitrate, and far-field microphones reduce ASR accuracy.
- **No cross-recording speaker memory.** Every job's speaker labels are local and unrelated to any other job's labels.
- **No speaker identification.** Section 10 describes what it would take to add it — not implemented today.
- **Parakeet performance is a target, not a validated number** (Section 13) — real production benchmarking is still pending.
- **Jobs are in-memory only** — a server restart loses all job state. A PostgreSQL schema exists but is not yet wired in.
- **A hung job still blocks the queue.** The stuck-job watchdog (Section 14) now fails the job so the UI reaches a terminal state, but it cannot kill the wedged worker thread — the single-worker executor stays occupied until the process is restarted.
- **GPU behaviour is reasoned, not yet measured.** The VRAM sizing, cross-engine GPU lock, and OOM fallbacks were verified by code audit and unit tests with mocked CUDA failures — not against a real GPU under genuine memory pressure. See Section 13.

---

## 20. Interview Cheat Sheet

**Q: What does your system do?**
It converts audio recordings into speaker-labeled transcripts, and then extracts structured meeting artifacts — summary, action items, decisions, questions, timeline — from that transcript.

**Q: How does speech become text?**
An acoustic model converts the waveform into learned tokens, which decode into text. Modern decoders (like Parakeet's TDT) emit a timestamp per token as part of decoding, so timestamps come for free rather than needing a separate step.

**Q: What is ASR?**
Automatic Speech Recognition — it converts speech to text automatically, without a human transcriber or hand-built rules. It has no concept of speaker identity; that's a separate problem, solved by a separate model.

**Q: What does audio preprocessing actually do in your system?**
It converts whatever format is uploaded into 16 kHz mono 16-bit PCM WAV via a single ffmpeg call — nothing more. 16 kHz because both ASR models were trained at that rate and speech content lives below 8 kHz anyway (Nyquist); mono because stereo speech is usually duplicated across channels and doubles compute for no accuracy benefit; PCM to avoid feeding lossy-codec artifacts into the model.

*Follow-up: Do you normalize loudness or trim silence?*
No — this project's preprocessing step is strictly format conversion. Silence-skipping happens downstream, inside WhisperX's VAD filter and implicitly inside pyannote's segmentation model, not as an explicit preprocessing stage.

**Q: Why Parakeet?**
Parakeet's TDT decoder gives native word-level timestamps without a separate alignment pass, it's GPU-optimized, and it's our current production ASR path, running via NVIDIA's NeMo toolkit.

**Q: Why not just use Whisper?**
Whisper is still supported as a fallback backend, but it needs a separate wav2vec2 alignment model to get word-level timestamps, and it doesn't diarize on its own. Parakeet gives us timestamps natively, which simplifies the pipeline.

**Q: What does TDT actually do?**
TDT (Token-and-Duration Transducer) predicts a token and a duration together at each decoding step, instead of stepping one audio frame at a time like a standard RNN-Transducer. The duration prediction lets the decoder skip ahead efficiently, and because it's predicting *how long* each token lasted, word-level timing comes directly out of decoding — no separate alignment model needed.

**Q: What is forced alignment, and why doesn't Parakeet need it?**
Forced alignment takes text that's already known to be correct and figures out exactly when each word was spoken in the audio — that's what wav2vec2 does for WhisperX, as a second model after Whisper produces the text. Parakeet doesn't need this because its TDT decoder produces timing as a first-class part of decoding, not as a post-hoc alignment problem.

**Q: What is NeMo?**
NVIDIA's toolkit/framework for loading and running NVIDIA's speech models — it's the framework, Parakeet is the model that runs inside it.

**Q: What is a Conformer encoder?**
The architecture that turns audio into the internal representation the decoder reads. It combines self-attention (good at long-range context) with convolution (good at local acoustic patterns like phoneme structure), which is why it's the standard high-accuracy ASR encoder.

**Q: What is WhisperX?**
A wrapper around Whisper that adds forced alignment (via wav2vec2) for accurate word-level timestamps — Whisper alone only gives segment-level (sentence-level) timestamps. It's not a diarization system — we still run pyannote separately for that even in the WhisperX path.

**Q: What Whisper model sizes are there, and what do they cost?**
Publicly, OpenAI's Whisper ranges from tiny (39M params, ~1GB VRAM) to large-v2/v3 (1.5B params, ~10GB VRAM at fp16) — bigger models trade speed for accuracy. This project defaults to `large-v3-turbo` and runs it through CTranslate2 with int8 quantization on CPU, which cuts the memory footprint substantially versus full fp16 — that's what makes running it without a dedicated Whisper GPU practical.

**Q: What is speaker diarization?**
The process of determining who spoke when, without knowing who anyone actually is — it detects speech regions, extracts a voice embedding per region, clusters similar embeddings together, and labels each cluster with an anonymous ID like SPEAKER_00.

**Q: How do you know Speaker 1 and Speaker 2?**
ASR gives us words with timestamps; diarization gives us speaker segments with timestamps, independently. We match each word to whichever diarization segment overlaps it the most (not nearest-timestamp — actual time overlap), then map the raw speaker ID to a display label like "Speaker 1."

*Follow-up: Why max-overlap and not nearest-timestamp?*
A word has duration, and can straddle a speaker change. Comparing which speaker covers the largest share of the word's actual duration is more accurate than comparing to a single point like the word's midpoint.

**Q: What are speaker embeddings?**
A fixed-length numeric vector produced by a neural network that summarizes a voice's acoustic characteristics — pitch, timbre, resonance — as a point in a high-dimensional space, trained so that same-speaker vectors land close together and different-speaker vectors land far apart.

**Q: How are embeddings clustered into speakers?**
Embeddings that are close together by a similarity metric (e.g. cosine similarity) get grouped into the same cluster; each cluster becomes one anonymous speaker.

**Q: What is the embedding dimension?**
That's internal to the pretrained `pyannote/speaker-diarization-3.1` pipeline this project calls — it isn't configured or hardcoded anywhere in this codebase, so I'd check pyannote's model card for the pinned version rather than quote a number from memory. The honest answer in an interview is to say exactly that, not guess a number.

**Q: Does diarization know the person's real identity?**
No. It only clusters voices into anonymous groups. There's no name matching or voiceprint enrollment in this system.

*Follow-up: How would you add real speaker identification?*
Enroll each known speaker with a reference voice sample, embed it with the same model diarization uses, store that reference embedding, and at inference time compare each diarized cluster's embedding against the stored references by similarity — accepting a name match only above a confidence threshold, with an "unknown speaker" fallback below it. Not implemented today.

**Q: How do you handle overlapping speech?**
This is a known limitation, not a solved problem — pyannote's clustering and the max-overlap word-assignment step both degrade when two people talk simultaneously, since a word can genuinely belong to two overlapping voices at once. It isn't specifically engineered around in this project.

**Q: Why are you using your own pyannote pipeline instead of WhisperX's built-in diarization?**
Two reasons: a confirmed dependency-version conflict between WhisperX's pinned pyannote/torch versions and what this project needs, and an architectural preference — keeping ASR and diarization as separate, independently swappable components that can run in parallel on different hardware (GPU for ASR, CPU for diarization). We don't claim it's more accurate than WhisperX's own diarization — we haven't measured that; the reasons are dependency isolation and parallelism, not accuracy.

**Q: Why do you chunk long recordings?**
Longer audio means a longer sequence into the model, which means more GPU memory. We split long recordings into VRAM-sized chunks so memory use stays bounded and predictable regardless of recording length.

**Q: How do you choose chunk size?**
It's picked automatically from detected free VRAM using a lookup table — smaller GPUs get shorter chunks (e.g. 90s under 4GB free), larger GPUs get longer chunks (up to 1320s at 24GB+), trading fewer round-trips for more memory headroom needed.

**Q: What happens at chunk boundaries?**
Each chunk is padded with 0.5 extra seconds of audio on both sides so a boundary word is fully audible to the model in at least one chunk, but only words whose start time falls inside the chunk's true "nominal" (non-padded) window are kept — so each word is counted exactly once.

**Q: Why overlap?**
Without it, a word spoken right at a chunk boundary can get cut in half acoustically, and the model may transcribe it incorrectly or not at all in either chunk.

**Q: How do you remove duplicates at chunk boundaries?**
The nominal-vs-padded window rule: a word is only kept if its start timestamp falls within the chunk's nominal (owned) window, even though the model saw a larger padded window. This guarantees exactly-once ownership per word.

**Q: What happens if a chunk fails?**
If it's an out-of-memory error, we retry with a smaller chunk (halved size), up to a retry limit. For any other error, we clean up that chunk's temp files and propagate the failure — we don't silently drop it.

**Q: You run transcription and diarization concurrently for speed — doesn't that risk OOM if they're both on the same GPU?**
It did, and that was a real production issue on smaller cards: each engine's memory budget was individually safe, but computed without knowing the other was about to allocate too. The fix is a shared process-wide lock (`GPU_COMPUTE_LOCK`) that both engines acquire around their actual GPU compute — whichever grabs it first finishes before the other starts. The two threads still launch together so CPU-side work overlaps, but GPU kernels never run simultaneously when both engines are on CUDA. When diarization is on CPU instead (small/no GPU), the lock is a no-op and both stages run genuinely in parallel.

*Follow-up: Why a lock instead of a smarter scheduler?*
Because the actual constraint is simple — don't let two GPU-heavy stages allocate at the same instant — and a single lock solves exactly that with no added moving parts. A scheduler would be solving a problem we don't have (more than two GPU consumers, or a need to prioritize between them).

**Q: How do you prevent GPU OOM?**
Chunk sizes are picked from detected free VRAM up front. If a chunk still OOMs, we automatically halve it and retry, up to a retry limit. We also clear CUDA cache and run garbage collection between chunks.

**Q: What are the limitations?**
Diarization isn't perfect, especially with overlapping speech; there's no real speaker identity, only anonymous labels; audio quality strongly affects accuracy; we don't yet have a validated real-world performance benchmark for the Parakeet+GPU path; and jobs are held in memory only, so a server restart loses job state.

**Q: What would you improve next?**
Run and record real production benchmarks on the target GPU, wire the existing PostgreSQL schema into the live job store so jobs survive a restart, move pipeline execution into a subprocess so a wedged job can actually be killed rather than only marked failed, and — if the product needs it — build out speaker identification on top of the existing diarization embeddings.

---

## 21. Further Reading (Anchors)

Kept short and self-contained on purpose — the sections above are meant to stand alone without needing these, but they're here for deeper follow-up:

- [Whisper paper](https://cdn.openai.com/papers/whisper.pdf) — ASR model and training data
- [WhisperX paper](https://arxiv.org/abs/2303.00747) — forced alignment for word-level timestamps
- [wav2vec 2.0 paper](https://arxiv.org/abs/2006.11477) — the alignment model WhisperX uses
- [Conformer paper](https://arxiv.org/abs/2005.08100) — the encoder architecture behind Parakeet
- [RNN-Transducer overview](https://arxiv.org/abs/1211.3711) — the decoder family TDT extends
- [NVIDIA NeMo documentation](https://docs.nvidia.com/nemo-framework/) — the framework Parakeet runs in
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — the diarization library used
- [Speaker diarization survey](https://arxiv.org/abs/2101.09624) — embeddings, clustering, the field in general
- [Voice Activity Detection](https://en.wikipedia.org/wiki/Voice_activity_detection) — how silence/speech regions are detected
- [Server-Sent Events documentation](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — how live progress streaming works
