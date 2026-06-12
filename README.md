# STT Models & Voice Personalization — June 2026

> Status: **ACTIVE** · Last verified 2026-06-11
>
> Standalone publication of research produced for [MacParakeet](https://github.com/moona3k/macparakeet), a local-first macOS dictation and meeting-transcription app. References to "the app," CLI commands, integration seams, and ADRs refer to that codebase.
>
> The fork's full `docs/research/` directory is mirrored in [`research/`](research/), including the source file of this README ([`stt-models-and-voice-personalization-2026-06.md`](research/stt-models-and-voice-personalization-2026-06.md)).

Research into (a) whether better open-weight STT models exist for MacParakeet
than the current lineup — Parakeet TDT 0.6B v3 (default) / v2 (English opt-in)
via FluidAudio CoreML, Nemotron 3.5 ASR Beta, WhisperKit
`large-v3-v20240930_turbo` — and (b) practical paths to personalize STT to a
single speaker's voice and phrasing. Supersedes the STT sections of
[open-source-models-landscape-2026.md](research/open-source-models-landscape-2026.md)
(HISTORICAL, Feb 2026 snapshot).

**Method:** two adversarially-verified deep-research passes plus a targeted
primary-repo verification pass (all 2026-06-11). Round 1: model landscape
(115 claims extracted, top 25 verified by independent 3-vote panels → 21
confirmed, 4 refuted). Round 2: SpeechAnalyzer + personalization evidence
(119 claims → 23 confirmed, 2 refuted). Round 3: direct verification of
fine-tune toolchains, GPU costs, FluidAudio vocabulary internals, and LLM
post-correction literature against primary repos/papers. Every figure below
survived cross-checking or is explicitly flagged as vendor-only/unverified.

**Reader context:** scoped for an English-only speaker on an M5 Pro / 64 GB /
macOS 26.5, optimizing both dictation latency and long-form accuracy; cloud
GPU training acceptable; inference strictly on-device (ADR-001/002).

---

## TL;DR

1. **The cheapest accuracy win is already shipped: use Parakeet v2 for
   English.** v2 beats the v3 default on English in both the independent Open
   ASR Leaderboard (6.05% vs 6.32% avg WER) and FluidAudio's own CoreML
   benchmarks (2.1% vs 2.5% LibriSpeech test-clean); FluidAudio's docs say
   verbatim *"Use v2 if you only need English."* Zero engineering — the
   `parakeet-model` preference exists today.
2. **The standout new engine candidate is FluidAudio's Nemotron Speech
   Streaming EN 0.6B** — vendor-benchmarked *on this exact hardware* (M5 Pro,
   macOS 26.5) at 2.28% WER / 65× RTFx at the 1120 ms latency tier. True
   streaming for latency-sensitive dictation, same FluidAudio SDK and cache
   mechanics as the shipped Nemotron Beta engine.
3. **The English accuracy frontier moved to Conformer+LLM-decoder models**
   (IBM Granite Speech 4.0 1B at 5.52%, NVIDIA Canary Qwen 2.5B at 5.63%) —
   8–12× slower than Parakeet, reachable on Apple Silicon today only via the
   **mlx-audio** runtime, which also happens to be the most practical seam
   for loading a personally fine-tuned checkpoint.
4. **Acoustic fine-tuning to a typical native-English voice has weak expected
   ROI.** The published single-speaker evidence (Project Euphonia, 432
   speakers) shows gains scale with how *bad* the baseline is — transformative
   for severely impaired speech (median WER ~89%→13%), near-zero for typical
   speakers. For a voice already at ~2–5% WER, personalization value
   concentrates in **vocabulary**, where FluidAudio's CTC-word-spotter
   boosting (spec'd in this repo but **not yet wired** — see §7.1) and a
   documented Whisper-LoRA→WhisperKit path (§6.2) are the practical levers.
5. **Do not adopt Apple SpeechAnalyzer as an engine.** The only rigorous
   head-to-head measured it less accurate (14.0% vs 11.7% WER on earnings22)
   *and* ~5× slower than Parakeet v2 on-device, and its vocabulary hooks
   exist only on its weaker `DictationTranscriber` module (§3).
6. **Before any engine or fine-tune decision: benchmark on your own dictation
   corpus** — the app already stores (16 kHz WAV, raw transcript, corrected
   text) per dictation. §8 gives the recipe; §9 ranks the roadmap.

---

## 1. Current baseline (what MacParakeet ships today)

| Engine | Model | English WER | Runtime | Role |
|--------|-------|-------------|---------|------|
| Parakeet (default) | TDT 0.6B **v3** | 6.32% leaderboard avg · 2.5% LS test-clean (FluidAudio CoreML) · 4.5% AA-WER real-world | FluidAudio CoreML/ANE, ~155× RTFx | All three modes |
| Parakeet (opt-in) | TDT 0.6B **v2** | **6.05%** leaderboard avg · **2.1%** LS test-clean | FluidAudio CoreML/ANE, ~146× RTFx | English-only opt-in |
| Nemotron Beta | nemotron-multilingual-1120ms | n/a (multilingual streaming) | FluidAudio CoreML | Opt-in Beta |
| WhisperKit | large-v3-v20240930_turbo | ~7.75% leaderboard avg (Whisper-turbo class) | WhisperKit CoreML | Non-Parakeet languages |

The independent Artificial Analysis AA-WER v2 benchmark (real-world audio:
agent calls, parliamentary speech, earnings calls) puts Parakeet v3 at 4.5%
WER — roughly 2× its clean-LibriSpeech figure, a useful calibration that real
dictation/meeting WER is always worse than test-clean numbers. (Two adjacent
AA claims — that v2 scores *worse* than v3 on AA-WER, and that Voxtral leads
open-weight AA-WER — were **refuted 0-3** in verification; see §4.)

## 2. Landscape since the February 2026 snapshot

**Verified (3-0) state of the Open ASR Leaderboard, 27 Mar 2026 snapshot:**

| Model | Params | Avg WER | RTFx (A100) | License | Apple Silicon path |
|-------|--------|---------|-------------|---------|--------------------|
| IBM Granite Speech 4.0 1B (Mar 2026) | 1B | **5.52%** | 280 | Apache 2.0 | mlx-audio |
| NVIDIA Canary Qwen 2.5B | 2.5B | 5.63% | 418 | CC-BY-4.0 | mlx-audio |
| Microsoft Phi-4-Multimodal | — | 6.02% | — | MIT | (not pursued) |
| **Parakeet TDT 0.6B v2** | 0.6B | 6.05% | **3,390** | CC-BY-4.0 | **FluidAudio CoreML (shipped)** + mlx-audio |
| Parakeet TDT 0.6B v3 | 0.6B | 6.32% | 3,330 | CC-BY-4.0 | FluidAudio CoreML (shipped) + mlx-audio |
| Voxtral Small 24B | 24B | 6.62% | 54 | Apache 2.0 | mlx-audio (4-bit) |
| Whisper Large v3 | 1.55B | 7.44% | 146 | MIT | WhisperKit (shipped) |

Key structural fact from the leaderboard maintainers: **Conformer-encoder +
LLM-decoder models lead accuracy; CTC/TDT decoders (Parakeet's family) trade
a small WER penalty for 10–100× higher throughput.** RTFx here is measured on
NVIDIA A100 batch harnesses — it does *not* transfer to Apple Silicon, where
the only on-device measurements that exist are vendor self-benchmarks.

**Other verified entrants:**

- **Moonshine v2** (Useful Sensors, Feb 2026, MIT): English-only, three sizes
  (33M/123M/245M); Medium self-reports 2.08%/5.00% LS clean/other, 6.65%
  multi-benchmark avg, streaming-first architecture, 258 ms response latency
  on an M3 — but its only Apple Silicon runtime is C++/ONNX **on CPU** (the
  paper never mentions CoreML, ANE, Metal, or MLX). Interesting for
  ultra-low-latency experiments; accuracy still trails in-app Parakeet v2.
- **Voxtral Realtime 4B** (Mistral, Feb 2026, Apache 2.0): streaming STT with
  80–2400 ms adjustable latency, pre-converted at
  `mlx-community/Voxtral-Mini-4B-Realtime-2602` (4-bit + fp16). Its interest
  is streaming latency, not accuracy (the 24B Voxtral underperforms Parakeet
  on English; the Realtime 4B is unbenchmarked on-device).
- **Microsoft VibeVoice-ASR** (Jan 2026, MIT, 9B): ASR + diarization +
  timestamps on up to 60-minute audio, in mlx-audio. Heavy; meeting-mode
  curiosity only.

### 2.1 The big new engine candidate: Nemotron Speech Streaming EN 0.6B

FluidAudio's catalog now includes an **English-specific streaming engine**
(`FluidInference/nemotron-speech-streaming-en-0.6b-coreml`, FastConformer-
RNNT, Apache 2.0, int8 encoder on ANE), vendor-benchmarked **on an Apple
M5 Pro running macOS 26.5**:

| Latency tier | WER (LS test-clean) | RTFx |
|--------------|---------------------|------|
| 560 ms | 2.28% | 42.1 |
| **1120 ms** | **2.28%** | **65.0** |
| 2240 ms (default) | 2.46% | 93.6 |

That matches or beats Parakeet v3's English WER *while streaming* — the
architecture dictation wants (partial results while you speak; near-zero
finalize cost on hotkey release). Verification confirmed the model is wired
into the same FluidAudio SDK MacParakeet already uses (registry entries
`nemotronStreaming560/1120/2240` in `ModelNames.swift`, a
`NemotronStreamingAsrManager` Swift API) — same runtime, download, and cache
mechanics as the shipped `nemotron-multilingual-1120ms` Beta engine.

Caveats: vendor self-benchmark, 100-file test-clean subset (the Parakeet rows
use 2,620 files), and the HF card notes the CoreML streaming export degrades
WER vs its ~1.79% PyTorch reference.

### 2.2 mlx-audio: the second runtime that changes the integration calculus

`Blaizzy/mlx-audio` (~7.3k stars, v0.4.4 released 2026-06-06) was verified —
in code, not just README — to run:

- **Parakeet TDT v2 and v3** with streaming and token-level timestamps,
- **Nemotron 3.5 ASR** in cache-aware streaming
  (`mlx-community/nemotron-3.5-asr-streaming-0.6b`),
- **Canary, Granite Speech, Voxtral (+ Realtime), Moonshine, Distil-Whisper,
  Qwen3-ASR, VibeVoice-ASR, MMS** — real implementation directories, with
  3/4/6/8-bit quantization.

Two properties matter for MacParakeet:

1. It is the **only working Apple Silicon path to the accuracy-frontier
   models** (Granite, Canary Qwen) today.
2. Its `load_model()` accepts **arbitrary HuggingFace repos or local paths** —
   the concrete seam for loading a *personally fine-tuned* checkpoint, which
   FluidAudio's fixed-repo catalog does not offer.

Caveats: MLX runs on CPU/GPU via Metal, **not the ANE**; per-model on-device
quality/RTF on the M5 Pro is unbenchmarked; it's a Python-ecosystem runtime,
so integration needs a packaging strategy (embedded runtime or sidecar
process — cf. the pre-FluidAudio parakeet-mlx daemon the app deliberately
migrated away from in ADR-007; `FluidInference/swift-parakeet-mlx` exists as
a Swift-native alternative for the Parakeet family).

### 2.3 Fine-tune conversion seam: FluidAudio's möbius pipeline

FluidAudio publishes its model-to-CoreML conversion pipeline as the open
Apache-2.0 repo **`FluidInference/mobius`** (last pushed 2026-06-10), which
the FluidAudio README points to for converting custom models. Verified
contents: `models/stt/parakeet-tdt-v3-0.6b/coreml/` with `convert-parakeet.py`,
`quantize_coreml.py`, `compile_modelc.py`, Torch-vs-CoreML parity validation,
and parallel directories for parakeet-v2, nemotron-streaming, and canary-1b-v2.
`convert-parakeet.py` defaults to pulling `nvidia/parakeet-tdt-0.6b-v3` but
exposes a **`--nemo-path` flag loading any `.nemo` checkpoint** via
`EncDecRNNTBPEModel.restore_from()`.

So a same-architecture fine-tuned Parakeet checkpoint **might convert without
code changes** — but there is *no documented custom-checkpoint support*, all
CoreML components export against a fixed 15-second audio window, and the
fine-tune→FluidAudio path is not turnkey (verification vote 2-1 on the
details). It must be proven hands-on before being load-bearing; the
mlx-audio arbitrary-checkpoint seam is the lower-risk alternative.

## 3. Apple SpeechAnalyzer / SpeechTranscriber (macOS 26): not a win

All claims verified against Apple's docs, the WWDC25 session 277 transcript,
and the Argmax benchmark post.

**Usability: yes.** `SpeechAnalyzer`/`SpeechTranscriber`/`DictationTranscriber`
are public, non-beta, third-party-usable APIs on macOS 26.0+ (Apple Silicon).
Model assets are system-managed via `AssetInventory` — no app download/storage
cost, inference runs outside the app's memory space, system-updated. Shipping
third-party apps (Dictato, FluidInference/swift-scribe, the `yap` CLI) already
use them. Requirements: `NSSpeechRecognitionUsageDescription` + mic permission.

**Accuracy/speed: behind Parakeet.** The only rigorous head-to-head (Argmax,
June 2025, M4 Mac mini, macOS 26 Beta Seed 1, ~12 h earnings22 subset):
SpeechTranscriber **14.0% WER at 70× speed** vs Parakeet v2 **11.7% WER at
359×** — less accurate *and* ~5× slower. Apple publishes no WER/speed numbers
at all (verified: zero quantitative figures in the WWDC session). Caveats:
Argmax is a competing vendor; Beta Seed 1 likely understates today's 26.5
models (9to5Mac measured ~8% on a different single-video test).

**The customization trap.** The flagship `SpeechTranscriber` (the accurate
module) exposes **no** vocabulary hook — no contextual strings, no custom LM
(verified against its full API surface; community reports confirm it ignores
`AnalysisContext.contextualStrings`). All documented customization —
`contextualStrings` biasing (≤100 phrases, 1–2 words each) and full custom
LMs via `SFCustomLanguageModelData` (X-SAMPA pronunciations, weighted phrase
counts; a corrected-transcript corpus could feed this directly) — routes
through **`DictationTranscriber`**, which Apple documents as using *"the same
speech-to-text machine learning models as system dictation… or
SFSpeechRecognizer"* — the older, weaker engine. On Apple's stack the
accurate model takes no vocabulary, and the customizable model is the weak
one.

**Verdict:** no reason to adopt while in-app Parakeet v2 is both more
accurate and faster. Worth a one-day re-benchmark when Apple ships a major
Speech update (their 26.x models are improving), and `SFCustomLanguageModelData`
remains an interesting reference design for corpus-driven LM personalization.

## 4. Refuted claims (do not cite)

Recorded so future research doesn't re-import them as fact (all 0-3 votes):

1. Cohere `cohere-transcribe-03-2026` / Qwen3-ASR leading the Open ASR
   Leaderboard.
2. Parakeet v2 scoring 6.4% AA-WER (worse than v3) on Artificial Analysis.
3. Voxtral leading open-weight AA-WER at 2.8%.
4. FluidAudio's Parakeet v3 CoreML build supporting Japanese.
5. "SpeechAnalyzer exposes no custom-vocabulary mechanism" (hooks exist — on
   `DictationTranscriber`; see §3).
6. arXiv:2110.04612 as evidence of *typical-speaker* personalization (all 195
   participants had speech impairments).

## 5. Is single-speaker fine-tuning worth it? The evidence says: not for a typical voice

The strongest published single-speaker results are on **disordered speech**
(Google Project Euphonia, peer-reviewed, 432 speakers each recording ≥300
utterances ≈ 0.25–2 h of audio): personalized RNN-T models dropped median WER
for severely impaired speakers **from ~89% to 13%** on short phrases.

The decision-relevant pattern, verified 3-0 with an adversarial
counter-evidence search that found nothing for typical speakers:
**gains scale with how bad the baseline is.** In the same study, the
typical-severity subgroup (n = 71) showed **near-zero WER deltas**; moderate/
severe speakers gained a median 49% accuracy. Adjacent literature agrees
(residual adapters: up to 77% relative WER reduction for atypical speech, 31%
for *accented* — arXiv:2109.06952). **No credible source shows large
acoustic-fine-tuning gains for a typical native English speaker already at
~2–5% WER.** For such a user, expected fine-tune ROI is low and concentrated
in domain-vocabulary recovery — which cheaper LM-level methods also target
(§7).

Where fine-tuning *does* make sense: strong accent, atypical speech, a
persistently jargon-heavy domain the vocabulary tools can't cover, or a
measured personal-corpus WER far above the model's benchmark norms (§8 tells
you which case you're in).

## 6. Fine-tuning toolchains and costs (verified against primary repos)

### 6.1 NeMo → Parakeet (CoreML via möbius): plausible, not turnkey

- NVIDIA's official tutorial ("How to Fine-Tune a Riva ASR Acoustic Model
  with NVIDIA NeMo") uses NeMo's `speech_to_text_finetune.py` + JSON-lines
  manifests, single-GPU bf16. Nuance: it demonstrates on a FastConformer
  *hybrid* checkpoint ("Parakeet family"), not `parakeet-tdt-0.6b` itself;
  the same workflow is the supported path for TDT checkpoints.
- Data guidance is thin: "<50 hrs → reuse the pretrained tokenizer." For very
  small datasets NVIDIA's own recommendation is **adapters, not full
  fine-tuning**: the NeMo ASR Adapters tutorial demonstrates adaptation "with
  minimal amounts of data, even just 30 minutes!" in ~10 minutes on one GPU,
  and warns full fine-tuning on small data degrades the model.
- Then: möbius `convert-parakeet.py --nemo-path` (§2.3) → CoreML →
  pre-populate FluidAudio's cache. **Unproven end-to-end** — and note the
  möbius scripts target standard checkpoints; adapter-augmented models would
  need merging first.

### 6.2 Whisper LoRA → WhisperKit: the documented, proven path

- HF's canonical recipe (huggingface.co/blog/fine-tune-whisper) demonstrates
  fine-tuning on **~8 hours** of training data; official PEFT examples
  fine-tune whisper-large-v2 with LoRA + INT8 **on a free Colab T4 (16 GB)**.
- **whisperkittools explicitly supports custom checkpoints**: the
  `whisperkit-generate-model` CLI's `--model-version` accepts "a Hugging Face
  model hub name … [or] a local directory containing the model files," and
  the WhisperKit README says verbatim it "lets you create and deploy your own
  fine tuned versions of Whisper in CoreML format," loaded via
  `WhisperKitConfig(model:…, modelRepo: "username/your-model-repo")`.
- This drops into MacParakeet's existing WhisperKit engine via the
  `downloadBase` custom-model-directory seam with modest plumbing.
- Trade-off: even fine-tuned, Whisper-turbo-class English WER starts behind
  Parakeet v2, so a personal LoRA must beat that gap on *your* corpus first.

### 6.3 Local MLX fine-tuning: real, but young

`ml-explore/mlx-examples` whisper and `mlx-audio` are **inference-only**
(verified). The one real option is **`ARahim3/mlx-tune`** (1.3k stars,
created 2026-01, active, single maintainer): self-reported **Stable** LoRA
STT fine-tuning of Whisper *and* Parakeet TDT (CTC/RNN-T/TDT losses) natively
on Apple Silicon, 16 GB+ recommended — the M5 Pro/64 GB qualifies easily.
Quality claims are the project's own; unbenchmarked independently.

### 6.4 Cost (verified mid-2026 prices)

A 0.6B-param ASR fine-tune over 1–5 h of personal audio is a 1–4 GPU-hour
job: RunPod A100 80 GB at **$1.39/hr** ≈ **$2–6**; H100 ≈ $3–13; $20 is a
generous ceiling. Whisper-LoRA fits a free/cheap Colab T4. **GPU cost is
negligible — the real costs are data prep and the conversion/validation
loop.**

## 7. Cheaper personalization than weight tuning

### 7.1 FluidAudio CTC-WS custom vocabulary — spec'd here, not yet wired

Correction to the MacParakeet repo's own assumptions: `spec/06-stt-engine.md` describes
FluidAudio `CustomVocabularyTerm` aliases, but **a grep of `Sources/` finds
zero references** — MacParakeet's shipping vocabulary feature is the
deterministic text-pipeline replacement, not FluidAudio biasing.

What the (verified) FluidAudio feature actually is: **acoustic-evidence-gated
post-decode rescoring** driven by a parallel CTC keyword spotter (NVIDIA's
CTC-WS method, arXiv:2406.07096) — stronger than find/replace (it only
rewrites when the audio supports the term), weaker than true decoder biasing.
Capabilities/limits (from `Documentation/ASR/CustomVocabulary.md`):
aliases→canonical normalization, weights, 1–50 terms "excellent" / up to 230
tested, terms ≥4 chars, stopwords skipped, vendor-claimed 99.4% dict recall.
Costs on Parakeet 0.6B: a separate ~97.5 MB CTC-110M encoder and ~2.7× RTFx
hit (155×→~26× — still far beyond realtime). **Parakeet TDT only — no
Nemotron support, degraded in streaming mode.**

This is the evidence-aligned personalization lever: typical-speaker error
concentrates in proper nouns/jargon, exactly what CTC-WS targets, fed
directly from the existing `custom_words` table.

### 7.2 WhisperKit prompt conditioning

`DecodingOptions.promptTokens` (prepended conditioning, ≈ `initial_prompt`)
and `prefixTokens` exist (verified in Configurations.swift). Indirect
vocabulary nudging only — no weighted-boost mechanism. Marginal value while
Whisper isn't the English engine.

### 7.3 Apple `SFCustomLanguageModelData`

First-party no-GPU custom-LM pipeline (X-SAMPA pronunciations, weighted
phrase counts, bulk template generators) that a corrected-transcript corpus
could feed directly — but its documented consumer is `DictationTranscriber`
(the weak engine, §3), and no published WER evaluation of the mechanism
exists. Reference design, not a current adoption target.

### 7.4 LLM post-ASR correction: evidence says skip it at this WER

The generative-error-correction literature is consistent (HyPoradise
NeurIPS'23, GenSEC SLT'24, Ma et al. 2024 — all verified from primary
sources): fine-tuned LLM correction produces 40–80% relative WER gains **on
weak baselines** (8–15% WER) and can even beat the N-best oracle — but on
strong baselines gains vanish: LibriSpeech clean 1.8%→1.7%, and **other
3.7%→3.8% (a regression)**, with documented meaning-drift/hallucination risk.
At Parakeet-class ~2–2.5% clean WER, expected gains are marginal-to-negative.
**No published work validates learning from a user's own correction history**
— that's an evidence gap, not an established technique. MacParakeet's AI
Formatter already occupies the post-ASR LLM slot for formatting; a cheap
experiment is enriching formatter prompts with the user's frequent
corrections, but it should be framed as formatting, not error correction.

## 8. Benchmark on your own corpus first (recipe)

Leaderboard deltas of 0.3–0.5% WER are smaller than the variance between
*your* microphone/voice/vocabulary and any benchmark. The app already stores
everything needed:

1. **Corpus**: `dictations` rows with `audioPath IS NOT NULL` (audio retention
   on) give (16 kHz WAV, `rawTranscript`, `cleanTranscript`) tuples;
   user-corrected rows are gold labels. Export:
   `macparakeet-cli history dictations --json`.
2. **Candidates**: per WAV, run
   `macparakeet-cli transcribe <wav> --engine parakeet --parakeet-model v2|v3`
   (and `--engine nemotron` / `--engine whisper`); run non-integrated
   candidates (Nemotron Streaming EN via a FluidAudio sample, Granite/Canary
   via mlx-audio, SpeechAnalyzer via a 20-line Swift CLI) outside the app.
3. **Score**: WER vs corrected text (`jiwer`/`werpy`), normalizing
   case/punctuation (dictation output passes through the text pipeline
   anyway). Track proper-noun errors separately — that's the vocabulary
   signal.
4. **Decide**: an engine or fine-tune must beat the incumbent **on this
   corpus** to justify adoption; a high personal WER vs benchmark norms is
   the one signal that would re-rank fine-tuning upward (§5).

## 9. Ranked adoption roadmap

| # | Item | Effort | Expected impact | Integration seam | ADR touchpoints |
|---|------|--------|-----------------|------------------|-----------------|
| 0 | **Personal eval harness** — script the §8 corpus benchmark; add proper-noun error breakdown | S | Decision-quality for everything below | CLI `transcribe`/`history` + a small WER script (could become a dev CLI subcommand) | none |
| 1 | **Default English users to Parakeet v2** (or at minimum surface a "best for English" nudge in Settings/onboarding) | S | +0.3–0.4% absolute WER, zero risk, shipped tech | existing `parakeet-model` preference | ADR-001 note (v2 default-for-English heuristic) |
| 2 | **Trial Nemotron Speech Streaming EN as a dictation engine** (flag-gated Beta) | M | Streaming partials + strong English WER on this exact hardware; biggest UX upside for dictation | FluidAudio SDK (`NemotronStreamingAsrManager`, registry `nemotronStreaming1120`) — same mechanics as shipped Nemotron Beta | ADR-001/ADR-016 amendments (new engine variant, lease semantics) |
| 3 | **Wire FluidAudio CTC-WS custom vocabulary** into the Parakeet path, fed from the existing `custom_words` table (acoustic-gated proper-noun recognition; keep the text-pipeline replacement as fallback) | M | The evidence-aligned personalization win for a typical speaker | `AsrManager.transcribe(_, customVocabulary:)` + ~130 MB CTC asset download; Parakeet-only | ADR-001 note; spec/06 already describes it |
| 4 | **Prototype an mlx-audio-backed 4th engine** for accuracy-frontier models (Granite 1B / Canary Qwen) in meeting/file mode; doubles as the custom-checkpoint loading seam | L | Up to ~0.5–0.8% leaderboard WER over Parakeet v2 for long-form; unverified on-device perf; packaging risk (Python sidecar vs swift-parakeet-mlx) | new `SpeechEnginePreference` case + `STTRuntime` dispatch (7-step seam) | ADR-001/016/021 amendments; revisits ADR-007's no-Python stance |
| 5 | **Personal fine-tune — only if #0 shows a real gap** (accent/jargon-heavy corpus): first choice Whisper-LoRA (free-tier GPU, documented whisperkittools conversion, loads via WhisperKit `downloadBase`); second choice NeMo adapter/fine-tune → möbius `--nemo-path` → CoreML (unproven end-to-end); local option mlx-tune (young) | M–L | Low expected ROI for a typical voice (§5); GPU cost trivial ($2–20), validation effort is the real cost | WhisperKit custom-model seam; FluidAudio cache pre-population | ADR-021 (custom Whisper model), ADR-002 (training data leaves device only by user action) |
| — | **SpeechAnalyzer engine: do not adopt** (slower + less accurate; vocab hooks only on its weak module). Re-benchmark on major macOS Speech updates | — | — | — | — |
| — | **LLM error-correction layer: do not build** (negative expected value at this WER; hallucination risk). Optional cheap experiment: feed frequent personal corrections into AI Formatter profile prompts | — | — | existing AI Formatter | — |

## Sources

**Leaderboards / benchmarks:** [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) · [leaderboard analysis blog](https://huggingface.co/blog/open-asr-leaderboard) · [leaderboard paper (arXiv:2510.06961v4)](https://arxiv.org/html/2510.06961v4) · [Artificial Analysis speech-to-text](https://artificialanalysis.ai/speech-to-text) · [Argmax: Apple and Argmax](https://www.argmaxinc.com/blog/apple-and-argmax) · [9to5Mac SpeechAnalyzer test](https://9to5mac.com/2025/07/03/how-accurate-is-apples-new-transcription-ai-we-tested-it-against-whisper-and-parakeet/)

**Runtimes / models:** [FluidAudio](https://github.com/FluidInference/FluidAudio) · [FluidAudio Benchmarks.md](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md) · [FluidAudio CustomVocabulary.md](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/ASR/CustomVocabulary.md) · [nemotron-speech-streaming-en-0.6b-coreml](https://huggingface.co/FluidInference/nemotron-speech-streaming-en-0.6b-coreml) · [parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) · [möbius](https://github.com/FluidInference/mobius) · [mlx-audio](https://github.com/Blaizzy/mlx-audio) · [Voxtral Realtime MLX](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) · [Moonshine v2 paper](https://arxiv.org/html/2602.12241v1) · [moonshine repo](https://github.com/moonshine-ai/moonshine) · [swift-parakeet-mlx](https://github.com/FluidInference/swift-parakeet-mlx)

**Apple Speech framework:** [SpeechTranscriber](https://developer.apple.com/documentation/speech/speechtranscriber) · [DictationTranscriber](https://developer.apple.com/documentation/speech/dictationtranscriber) · [AnalysisContext.contextualStrings](https://developer.apple.com/documentation/speech/analysiscontext/contextualstrings) · [SFCustomLanguageModelData](https://developer.apple.com/documentation/speech/sfcustomlanguagemodeldata) · [AssetInventory](https://developer.apple.com/documentation/speech/assetinventory) · [WWDC25 session 277](https://developer.apple.com/videos/play/wwdc2025/277/)

**Personalization evidence:** [Project Euphonia blog](https://research.google/blog/personalized-asr-models-from-a-large-and-diverse-disordered-speech-dataset/) · Green et al., Interspeech 2021 (DOI 10.21437/Interspeech.2021-1384) · [residual adapters (arXiv:2109.06952)](https://arxiv.org/abs/2109.06952) · [CTC-WS (arXiv:2406.07096)](https://arxiv.org/abs/2406.07096) · [HyPoradise (arXiv:2309.15701)](https://arxiv.org/abs/2309.15701) · [GenSEC (arXiv:2409.09785)](https://arxiv.org/abs/2409.09785) · [Ma et al. (arXiv:2409.09554)](https://arxiv.org/abs/2409.09554) · [Whisper TCPGen biasing (arXiv:2410.18363)](https://arxiv.org/abs/2410.18363)

**Fine-tune toolchains:** [Riva/NeMo Parakeet fine-tune tutorial](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-finetune-parakeet-nemo.html) · [NeMo ASR Adapters tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/asr_adapters/ASR_with_Adapters.ipynb) · [HF fine-tune-whisper](https://huggingface.co/blog/fine-tune-whisper) · [PEFT Whisper LoRA notebook](https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb) · [whisperkittools](https://github.com/argmaxinc/whisperkittools) · [WhisperKit](https://github.com/argmaxinc/WhisperKit) · [mlx-tune](https://github.com/ARahim3/mlx-tune) · [Lambda pricing](https://lambda.ai/pricing) · [RunPod pricing](https://www.runpod.io/pricing)
