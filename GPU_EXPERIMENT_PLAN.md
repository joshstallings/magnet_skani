# MAGNET GPU Experiment Plan (Open-Source, Custom CUDA)

## Detailed Summary of the Experiment

This experiment extends the current `magnet_skani` repository from a CPU-based `skani` speedup to a fully open-source GPU acceleration path focused on NVIDIA CUDA (with optional Tensor Core optimization only after correctness is proven). The main objective is to reduce total MAGNET runtime while preserving scientific output with original MAGNET pipeline.

The project is intentionally staged:
- Keep existing MAGNET behavior and outputs stable while introducing GPU modules behind explicit feature flags.
- Accelerate low-risk/high-impact internal compute first (Consensus ANI), then progressively move toward harder components (coverage aggregation, experimental GPU mapper).
- Maintain `skani` as the production clustering baseline until profiling proves ANI matrix construction is still the dominant bottleneck after other GPU work is complete.

Success criteria for this experiment:
- GPU and CPU runs produce equivalent cluster-level and final presence/absence outcomes.
- Stage-specific runtime reductions are measurable and reproducible with benchmark artifacts.
- Every GPU path has safe CPU fallback and deterministic validation mode.

Non-goals for the first execution cycles:
- No proprietary tools (for example, Parabricks) in the primary path.
- No default replacement of core production behavior until parity + performance gates pass.

---

## Detailed Step-by-Step Execution Plan

### Step 1: Establish a Reproducible Baseline
1. Add a benchmark at `scripts/bench_magnet.py`.
2. Define benchmark datasets (at least one ONT, one Illumina).
3. Capture per-stage wall clock and peak memory for:
   - skani pairwise ANI
   - alignment subprocess
   - samtools sort/index/coverage/consensus
   - `ani_summary` / `cal_ani`
4. Save baseline artifacts to `benchmarks/baseline/`:
   - timing JSON
   - environment metadata (GPU/CPU/driver versions)
   - output checksums and key output CSV snapshots
5. Freeze acceptance thresholds for correctness and speedup before coding GPU kernels.

### Step 2: Add Backend Abstractions and Runtime Controls
1. Extend CLI in `magnet.py` with:
   - `--compute-backend {cpu,cuda}`
   - `--gpu-device`
   - `--gpu-batch-size`
   - optional `--validate-gpu` flag
2. Add `utils/gpu/` package with:
   - `utils/gpu/__init__.py`
   - `utils/gpu/runtime.py` (device checks, fallback logic, stream/context setup)
   - `utils/gpu/kernels.py` (kernel wrappers and launch helpers)
3. Enforce backend dispatch contract:
   - If CUDA unavailable or kernel fails, fallback to CPU path automatically.
   - Log fallback reason in run output directory.
4. Add regression tests to confirm CPU default behavior remains unchanged when no GPU flags are used.

### Step 3: Implement the First CUDA Target (Consensus ANI)
1. Refactor `utils/ani.py` to isolate `cal_ani` core logic behind a backend interface.
2. CPU baseline path:
   - Keep existing logic as reference implementation.
3. CUDA path:
   - Convert sequence pairs into packed byte arrays.
   - Launch kernel to compute:
     - matched base count
     - effective compared length (excluding `N` positions)
   - Return ANI as `matched/effective`.
4. Add strict correctness tests:
   - random sequence tests
   - edge cases (`N`-heavy, no overlap, short contigs)
   - deterministic repeatability
5. Add performance tests comparing CPU vs CUDA ANI stage runtime.

### Step 4: Prototype GPU Coverage/Depth Aggregation
1. Keep current `samtools` output generation unchanged.
2. Refactor aggregation in `alignment_summary` (`magnet.py`) into pluggable reducer API:
   - CPU reducer (current pandas logic)
   - CUDA reducer (new)
3. CUDA reducer responsibilities:
   - parse coverage data into columnar numeric arrays
   - group by assembly index
   - compute breadth/depth/expected coverage intermediate values
4. Validate numerical parity against CPU reducer on small and medium test sets.
5. Record stage-specific speedups and overhead (including transfer time).

### Step 5: Add Experimental Custom CUDA Mapper Backend
1. In `utils/alignment.py`, introduce a backend-neutral mapper entrypoint:
   - `run_mapper(backend=..., mode=..., threads=..., ...)`
2. Keep existing CPU backends (`minimap2`, `bowtie2`) unchanged as references.
3. Add experimental CUDA mapper backend in phases:
   - MVP: ONT-focused seed-and-extend prototype on merged representative FASTA
   - Emit SAM-compatible records for existing downstream tooling
4. Validate mapper backend by comparing:
   - primary/secondary coverage statistics
   - downstream consensus ANI and presence/absence consistency
5. Only optimize for throughput after correctness parity is stable.

### Step 6: Keep skani as Production ANI Clustering Baseline
1. Continue using `compute_skani_pairwise_matrix*` in `magnet.py` for representative clustering.
2. Re-profile after Steps 3-5 to confirm if skani stage is now a dominant bottleneck.
3. If and only if still dominant:
   - start a separate experimental CUDA ANI sketching module
   - keep disabled by default
4. Require clustering-equivalence checks before any potential promotion.

### Step 7: Build Validation Gates and CI Criteria
1. Add unit tests:
   - GPU ANI parity
   - fallback behavior when CUDA not present
2. Add integration tests:
   - end-to-end output equivalence for representative selection and final calls
3. Add performance gates:
   - ANI stage speedup target for CUDA mode
   - end-to-end speedup target for at least one ONT dataset
4. Add stability gates:
   - no crashes under repeated runs
   - deterministic output with fixed inputs

### Step 8: Rollout Strategy
1. Release GPU features behind explicit opt-in flags only.
2. Publish benchmark results and parity tables per phase.
3. Iterate on kernel and batching parameters after correctness is locked.
4. Consider Tensor Core optimization pass only after:
   - exact/near-exact parity is proven
   - profiler indicates arithmetic bottleneck not memory/I/O bottleneck

---

## Immediate Implementation Order (Practical Next Sequence)
1. Baseline harness and metrics capture.
2. Backend flags and GPU runtime scaffolding.
3. CUDA consensus ANI implementation + parity tests.
4. GPU coverage aggregation prototype.
5. Experimental CUDA mapper MVP.
6. Re-profile and decide whether CUDA ANI sketching is warranted.

## Definition of Done for This Experiment
- End-to-end CUDA mode exists and is runnable with CPU fallback.
- Scientific outputs match baseline within pre-defined thresholds.
- Benchmarks show meaningful acceleration on at least one realistic dataset.
- Documentation clearly states maturity level of each GPU-accelerated stage.
