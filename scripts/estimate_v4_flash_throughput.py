"""Predict DeepSeek-V4-Flash-Base sampling wall-clock on H200 / B200.

Inputs are measured (data/*/run_meta.json), read off the checkpoint on DAIS
(config.json, model.safetensors.index.json), or vendor spec.  Two free
parameters, both swept: the fraction of peak HBM bandwidth an MoE decode step
achieves, and the host-side cost per prompt per round.
"""

# ---- checkpoint facts, read from DAIS --------------------------------------
WEIGHT_BYTES = 294_673_469_000  # model.safetensors.index.json total_size
HIDDEN, MOE_INTER = 4096, 2048
N_ROUTED, N_ACTIVE, N_SHARED = 256, 6, 1
LAYERS = 43

EXPERT_BYTES = LAYERS * N_ROUTED * 3 * HIDDEN * MOE_INTER  # fp8, 1 B/param
FIXED_BYTES = WEIGHT_BYTES - EXPERT_BYTES  # attention, shared experts, embeds, MTP
ACTIVE_PARAMS = LAYERS * (N_ACTIVE + N_SHARED) * 3 * HIDDEN * MOE_INTER + 6.3e9

# ---- the sampling loop, measured on the 4090 ------------------------------
STUDIES = {  # n, rounds/respondent, decode steps/round
    "pfander": (18_000, 78.1, 7.5),
    "voelkel": (6_203, 57.3, 7.5),
}
RETRY = 1.10  # rejection re-issues + structured fallbacks

#: Host-side ms per prompt per round in the current driver, which re-submits the
#: whole transcript as a *string* every round: tokenise ~7k tokens, hash them
#: into prefix-cache blocks, build the request, detokenise.  Backed out of the
#: 4090 run (515 ms/round, 15 prompts) by subtracting plausible decode time --
#: hence a band, not a number.
HOST_MS_PER_PROMPT = (6.7, 12.2, 16.6)

# ---- hardware --------------------------------------------------------------
GPUS = {"H200": (141, 4.8, 1.979), "B200": (180, 8.0, 4.500)}  # GB, TB/s, fp8 PFLOP/s
BW_BAND = (0.25, 0.40, 0.60)
FLOP_FRACTION = 0.35
LAUNCH_FLOOR_MS = 4.0  # ~650 kernels/step even with CUDA graphs


def step_ms(gpu, n_gpu, batch, bw_fraction):
    """Decode step: streaming weights vs expert GEMM compute, whichever binds."""
    _, tb_s, pflops = GPUS[gpu]
    touched = 1.0 - (1.0 - N_ACTIVE / N_ROUTED) ** batch
    read = FIXED_BYTES + touched * EXPERT_BYTES
    t_mem = read / (tb_s * n_gpu * 1e12 * bw_fraction) * 1e3
    t_flop = 2 * ACTIVE_PARAMS * batch / (pflops * n_gpu * 1e15 * FLOP_FRACTION) * 1e3
    return max(t_mem, t_flop, LAUNCH_FLOOR_MS), t_mem, t_flop


def fits(gpu, n_gpu):
    """88% of HBM for weights; the rest is activations, comms, CUDA graphs."""
    return GPUS[gpu][0] * n_gpu * 0.88 * 1e9 > WEIGHT_BYTES


def group_at_crossover(gpu, n_gpu, bw_fraction, cap=2048):
    """Biggest group still weight-bound.  Past it, more concurrency buys nothing."""
    best, g = 16, 16
    while g <= cap:
        _, t_mem, t_flop = step_ms(gpu, n_gpu, 4 * g, bw_fraction)
        if t_flop > t_mem:
            return best
        best, g = g, g * 2
    return best


def gpu_hours(study, gpu, n_gpu, bw_fraction, group=None):
    """Hours of GPU time.  ``group`` is fixed across the band so the band reads
    as a band: picking it per bandwidth fraction makes better bandwidth look
    *slower*, because compute binds sooner and the weight-bound group shrinks."""
    n, rounds, spr = STUDIES[study]
    group = group or group_at_crossover(gpu, n_gpu, BW_BAND[1])
    t_step, _, _ = step_ms(gpu, n_gpu, 4 * group, bw_fraction)
    steps = rounds * spr * RETRY
    return (n / group) * steps * t_step / 1e3 / 3600, group, t_step


def host_hours(study, ms_per_prompt):
    """Serial with the GPU today, and independent of how the groups are cut."""
    n, rounds, _ = STUDIES[study]
    return n * rounds * ms_per_prompt / 1e3 / 3600


print(f"weights            {WEIGHT_BYTES/1e9:7.1f} GB")
print(f"  experts (fp8)    {EXPERT_BYTES/1e9:7.1f} GB")
print(f"  everything else  {FIXED_BYTES/1e9:7.1f} GB")
print(f"active params      {ACTIVE_PARAMS/1e9:7.1f} B/token\n")

# compress_ratios: 2 layers at 0 (full MLA), 21 at 4 (8-position window),
# 20 at 128 (128-position window), + 1 at 0 for the MTP layer.
per_tok = 2 * 576 * 2
fixed_state = 20 * 128 * 512 * 4 + 21 * 8 * 2 * 512 * 4
print("KV per session: only 2 of 43 layers keep a full-length MLA cache, the rest")
print(
    f"are 8- or 128-position compressor windows -> {per_tok} B/token + "
    f"{fixed_state/1e6:.1f} MB fixed"
)
print(
    f"  7,500-token transcript: {(per_tok*7500 + fixed_state)/1e6:.0f} MB/session"
    f"  (Qwen2.5-7B: 229 MB)\n"
)

print("=" * 76)
print("GPU term only (host cost removed, i.e. the driver rewritten)")
print("=" * 76)
for study in STUDIES:
    n, rounds, spr = STUDIES[study]
    print(f"\n{study}: {n} respondents x {rounds*spr*RETRY:.0f} decode steps")
    print(
        f"{'config':>10} {'group':>6} {'ms/step':>8}   "
        + "   ".join(f"{int(f*100)}% bw" for f in BW_BAND)
    )
    for gpu in GPUS:
        for n_gpu in (2, 4, 8):
            if not fits(gpu, n_gpu):
                print(
                    f"{gpu+' x'+str(n_gpu):>10} {'--':>6} {'--':>8}   "
                    f"294.7 GB of weights do not fit in {GPUS[gpu][0]*n_gpu} GB"
                )
                continue
            cells = [f"{gpu_hours(study, gpu, n_gpu, f)[0]:5.2f} h" for f in BW_BAND]
            _, group, t_step = gpu_hours(study, gpu, n_gpu, BW_BAND[1])
            print(
                f"{gpu+' x'+str(n_gpu):>10} {group:>6} {t_step:>8.1f}   "
                + "   ".join(cells)
            )

print("\n" + "=" * 76)
print("Host term: serial with the GPU, identical on every configuration")
print("=" * 76)
for study in STUDIES:
    band = [host_hours(study, m) for m in HOST_MS_PER_PROMPT]
    print(
        f"  {study:>8}: {band[0]:5.2f} - {band[2]:5.2f} h   (central {band[1]:.2f} h)"
    )

print("\n" + "=" * 76)
print("Wall clock as the driver stands today = host + GPU")
print("=" * 76)
for study in STUDIES:
    h = host_hours(study, HOST_MS_PER_PROMPT[1])
    for gpu, n_gpu in (("H200", 4), ("H200", 8), ("B200", 8)):
        g = gpu_hours(study, gpu, n_gpu, BW_BAND[1])[0]
        print(
            f"  {study:>8} {gpu} x{n_gpu}: {h:5.2f} h host + {g:4.2f} h GPU = {h+g:5.2f} h"
        )

print("\nFor reference, the measured 4090 + Qwen2.5-7B runs:")
print("  pfander 13.40 h (18,000)   voelkel 3.28 h extrapolated to 6,203")
