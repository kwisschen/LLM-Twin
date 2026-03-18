# Key Architectural Decisions

These are production-grade decisions made throughout the LLM Twin project. Each entry explains the what, why, and tradeoffs. For interview-ready phrasing, see interview-narrative-log.md.

## Data Pipeline Decisions

### Embedding Model: nomic-embed-text-v1.5
Replaced all-MiniLM-L6-v2. 768-dim vs 384-dim doubles semantic precision for technical prose. MPS-accelerated on M1.

### Custom Token-Based Chunking
Replaced LangChain RecursiveCharacterTextSplitter with custom _split_by_tokens(). LangChain was re-instantiating the model without trust_remote_code on every call. Fix eliminated redundant model loading.

### Asymmetric RAG Indexing
Embed only the user prompt as the search vector; store the full prompt-response pair for generation context. Retrieval keyed to voice, full context available for generation.

### Atomic Conversation Chunking
One chunk per Q&A pair. Multi-turn conversations expanded into discrete retrieval documents. 68 ChatGPT conversations became 1,878 pairs.

### Per-File Repository Chunking
Original handler joined all repo files into one string -- large repos exceeded embedding model token limits. Restructured to split on #### file separator, then apply token-based chunking within each file. Result: 4,248 properly-sized chunks across 7 repos.

### Data Quality Gate: _is_garbled()
Four-check system in preprocessing layer: non-ASCII ratio (0.15), whitespace density, 4-gram repetition dominance, Shannon entropy bounds (1.5-5.8 bits/char). Category-aware: is_code=True skips whitespace check, raises entropy ceiling to 6.2.

### Binary Filter at ETL Boundary
Extension blocklist for 40+ binary formats plus null-byte sniffing for first 8KB. Changed file reads from errors='ignore' to errors='strict'. Reject at boundary, never silently transform.

### Category-Aware Dataset Preprocessing
Conversations skip extract_substrings() (already atomic Q&A pairs, 200-char floor). Articles/repos/posts use sentence-splitting (1000-char floor). Same modality-aware principle as is_code flag.

### Article Exclusion
All 76 articles in MongoDB were from book authors, not Christopher. Would have trained model to mimic someone else. Filtered at dataset level.

### ZenML Cache Invalidation
First SFT dataset had 80%+ garbled content because ZenML cached data-loading from pre-fix run. Added --no-cache when upstream data changes. Cache poisoning is stale feature stores in production ML.

### DPO Format Filter Retention
Kept filter_answer_format() for all categories including conversations. 25% of conversations would fail (lowercase starts, missing punctuation). SFT captured all content; DPO refines style on cleanest subset.

## Training Infrastructure Decisions

### LoRA over QLoRA
EdgeXpert has 128GB unified memory. Full BF16 LoRA avoids quantization noise that QLoRA introduces during forward pass. SageMaker 24GB A10G would force QLoRA. Match compute to quality constraint.

### NGC Container for Bleeding-Edge Hardware
GB10 is aarch64 ARM64 + CUDA 13.0 + Blackwell. Standard Python wheels do not support it. NGC containers ship pre-compiled binaries. Same approach as DGX systems.

### Eager Mode (Compile-Disable)
GB10 per-SM shared memory (101KB) breaks torch.compile auto-tuned Triton kernels (need 119KB+). Dual flags: UNSLOTH_COMPILE_DISABLE=1, TORCHDYNAMO_DISABLE=1. Trades ~10-20% wall-clock speed for zero quality impact.

### Container Recreation After Power Cycles
GPU device handles go stale after host reboot. docker start reuses old handles -- fails silently. Always docker rm + docker run with fresh GPU mapping.

### Docker Commit for Reproducibility
llm-training-ready image bakes in all packages. Container recreation after power cycle is instant instead of 5-min reinstall.

### SFT Adapter Preservation for DPO
Saved LoRA adapter (~336MB) separately from merged model (~16GB). DPO loads adapter on top of base model to continue refining. Merged model alone cannot separate base from SFT adjustments.

### Sequence Packing
19,124 samples compressed into 1,044 packed sequences. Multiple short samples concatenated into each 2048-token window. 18x step reduction with identical data coverage.

## Architecture-Level Decisions

### FTI Architecture
Feature pipeline -> Training pipeline -> Inference pipeline. Clean separation of concerns.

### Dispatcher Pattern
Routes document types to correct handlers. Additive by design -- new data types only require new handlers.

### Embedding Model as Versioned Decision
Tracked as an explicit architectural choice, not a tutorial default.

### Data Provenance Validation
All training data must be from Christopher. Validate provenance before training, not after.

### DPO Beta 0.5 over Default 0.1
DPO naturally pushes toward formal language because chosen responses in most datasets are more polished than rejected ones. For style imitation, this conflicts with the goal -- Christopher's writing includes conversational tone, contractions, and informal phrasing. beta=0.5 keeps the model closer to the SFT reference, preserving the voice learned in Phase 1. The handbook authors validated this through 20+ experiments. Result: eval accuracy 96%, natural output without GPT-isms.

### DPO Learning Rate 150x Lower than SFT
SFT lr=3e-4 teaches new capabilities (content, format, domain). DPO lr=2e-6 refines existing capabilities (style preference). Higher DPO learning rate causes catastrophic overwriting of SFT knowledge -- the model "forgets" content in pursuit of style. DPO is a refinement pass, not a retraining pass.

### Single Epoch for DPO
Multiple DPO epochs cause formal language drift -- each pass reinforces the formality bias inherent in preference data. One epoch provides sufficient preference signal (eval accuracy 94%→96%) without over-correction. Combined with high beta (0.5) and low lr (2e-6), this forms a "light touch" DPO strategy for style imitation.

### No Sequence Packing for DPO
SFT uses packing (18x step reduction). DPO cannot -- it processes prompt/chosen/rejected as aligned triples, computing log probability differences between chosen and rejected for the same prompt. Packing would break this alignment. DPO uses max_prompt_length + max_length split instead (1024/1024 of 2048 total).

### LoRA Reference Model Toggle
ref_model=None with LoRA. DPOTrainer toggles adapters on (trained policy) vs off (reference policy). Avoids loading two 8B models. On 128GB EdgeXpert, this optimizes speed not memory -- one model load instead of two.

### Pre-Run Trainer Smoke Test
Added after DPO warnings_issued bug. Signature inspection catches parameter renames but not runtime attribute errors from cross-package version mismatches (transformers ↔ peft ↔ unsloth ↔ trl). Instantiation smoke test with dummy data flushes out these errors in seconds instead of after a 20-minute model load.

Decision 25: GPT-4.1-mini as Evaluation Judge
Used GPT-4.1-mini (OpenAI) to judge Llama-based model outputs. Rationale: (1) External model family avoids family bias — OpenAI judging Llama is more objective than Llama judging Llama. (2) Consistent with data generation pipeline which also uses GPT-4.1-mini. (3) Cost-effective at ~$3 for 6,400 evaluations. (4) JSON structured output mode ensures reliable score parsing.

Decision 26: Two-Criteria Evaluation (Accuracy + Style)
Evaluated on accuracy (1-3) and style (1-3). Accuracy validates fine-tuning didn't catastrophically degrade knowledge — it's the safety check. Style is the actual success metric: did the model learn Christopher's informal, blog-appropriate voice? Two separate dimensions prevent a single aggregate score from hiding tradeoffs. The handbook used the same two criteria.

Decision 27: Transformers over vLLM for Evaluation Generation
vLLM would have been 5-10x faster for batch generation, but it overwrites NGC's custom PyTorch build and expects CUDA 12 library paths incompatible with the container's CUDA 13.x layout. Chose transformers model.generate() to preserve the training environment. Accepted ~8 hours per model instead of ~1 hour. vLLM will get its own dedicated container at the inference pipeline milestone. Training and serving containers must remain separate.

Decision 28: Re-merge Models Under Correct Transformers Version
Original merged models were pushed with transformers 5.2.0 but the evaluation container runs 4.57.6. The tokenizer incompatibility was caught and fixed (load from base model). The weight corruption was not caught until evaluation showed garbage outputs (://, template regurgitation). Re-merged SFT from base + adapter under 4.57.6 in-container. DPO-Merged was already loadable. Established rule: always merge and push from the same environment where models will be loaded.

Decision 29: DPO Beta Tuning (0.5 → 0.1)
Initial DPO with beta=0.5 degraded both accuracy (-0.32) and style (-0.31) vs SFT. Diagnosis: SFT already achieved 2.78/3.0 on style, leaving no headroom for DPO improvement. Beta=0.5 over-constrained the model near a local optimum that was slightly worse than SFT. Reduced to beta=0.1 (the standard default). Results: accuracy recovered from 2.05 to 2.31, style from 2.47 to 2.74. DPO still slightly below SFT, confirming the style saturation hypothesis. Best model for deployment: SFT.

Decision 30: Post-Training Quality Gate (New Operational Mandate)
Added mandatory quality gate after every training run: generate 10 sample outputs, review manually before proceeding. This was not in the original operational guardrails and should have been. The merged model corruption produced garbage that loss curves could not detect. The quality gate would have caught it in 5 minutes instead of days of generation and evaluation. Also added pre-generation verification checklist: script exists in container, fixes present, HF repo state checked, GPU available.

Decision 31: Schema Mismatch Handling for HuggingFace Hub
HF Hub rejects push_to_hub when the new dataset has different columns than the existing repo (e.g., generation pushes 5 columns, evaluation pushes 8). Added try/except handler that deletes the stale repo and re-pushes on ValueError. Applied to both generate_answers.py and evaluate_answers.py. This is a lasting solution, not a workaround — schema evolution is inherent to the pipeline where generation and evaluation add columns incrementally.

### Decision 32: vLLM Over TGI for Model Serving
vLLM chosen for the LLM microservice over TGI and raw transformers. OpenAI-compatible API out of the box — business microservice uses the standard `openai` Python client, making the inference backend swappable to any provider. Continuous batching + PagedAttention for production-grade throughput. Most widely adopted open-source inference engine. Raw transformers was proven too slow during evaluation (~8h for batch generation). TGI is viable but vLLM has broader community adoption and library support.

### Decision 33: Microservices Split — FastAPI on MacBook, vLLM on EdgeXpert
Business microservice (FastAPI + retrieval module) runs on MacBook. LLM microservice (vLLM) runs on EdgeXpert. Retrieval is CPU/IO-bound (OpenAI API calls, Qdrant queries, cross-encoder scoring). Generation is GPU-bound. Splitting avoids wasting GB10 GPU cycles on network IO. FastAPI communicates with vLLM via Tailscale (`<EDGEXPERT_IP>:8001`). Same architectural pattern as the handbook's SageMaker + FastAPI split, adapted for self-hosted infrastructure. The inference backend is a pluggable implementation detail — swapping vLLM on EdgeXpert to TGI on SageMaker is a config change, not a rewrite.

### Decision 34: SFT Model for Inference (Not DPO)
Deploying kwisschen/TwinLlama-3.1-8B-Merged (SFT) instead of the DPO variant. Evaluation showed SFT scored highest on style (2.78 vs 2.74) with comparable accuracy. DPO did not improve over SFT due to style saturation. Ship the model that scored best on the actual success metric.

### Decision 35: Separate vLLM Container (Permanent Policy)
vLLM runs in a dedicated Docker container (`vllm-inference`), never installed into the NGC training container. vLLM bundles its own PyTorch build (CUDA 12 paths) which silently overwrites NGC's custom PyTorch (CUDA 13.x). Identified during evaluation (Decision 27) and codified as permanent policy. Training and serving containers are permanently separate. Unlike the training container, `vllm-inference` was created from the official `vllm/vllm-openai` image with no additional pip installs, so `docker start` (not full recreation) is safe after container stops — GPU handle staleness only applies after host power cycles.

### Decision 36: ARM64 vLLM Compatibility (Native Image)
The official `vllm/vllm-openai:latest` image has ARM64 manifests, confirmed via `docker manifest inspect`. No fallback path needed. The GB10 (aarch64) runs the native image directly. This was the first thing investigated before any code changes — a non-negotiable gate in the spec.

### Decision 37: Prompt Template Alignment (Alpaca Format)
Inference prompt must exactly match the SFT training template: `Below is an instruction...### Instruction:\n{instruction}\n\n### Response:\n`. Mismatched templates produce garbage. Initial testing with a vague system instruction ("Write what the user asked you to") caused meta-commentary (model described what it would write instead of writing it). Fixed by using a directive persona: "You are Christopher Chen, a technical content creator. Respond directly to the following request." The Alpaca template structure was never changed — only the content slotted into `{instruction}`.

### Decision 38: Thread-Safety Fix for Embedding Singleton
The nomic SentenceTransformer singleton is not thread-safe. The original ContextRetriever ran embedding inside ThreadPoolExecutor workers, causing sporadic failures and empty results. Fixed by embedding all expanded queries serially on the main thread before dispatching parallel Qdrant searches. Qdrant HTTP calls are the IO-bound work that benefits from parallelism — embedding is CPU-bound and fast enough serially.
