"""
Monte Carlo experiment: Method A (avg probs) vs Method B (avg logits)

WHY MONTE CARLO?
  We cannot easily prove analytically when/why these two methods disagree --
  the interaction between softmax nonlinearity and argmax has no clean
  closed-form characterisation. Instead we simulate 50,000 random "news days"
  and let the empirical frequencies converge (by the law of large numbers) to
  the true probabilities. This is the Monte Carlo principle: replace a hard
  integral/proof with random sampling at scale.

WHAT WE SIMULATE:
  Each trial = one news day with T articles.
    - Each article has 3 random logits drawn from N(0, sigma).
    - Method A: softmax each article, average the probs, argmax.
    - Method B: average the logits, softmax, argmax.
  We record:
    - Did the methods agree on the dominant class?
    - How large was the biggest logit spike?
    - Did the most confident article (highest max-prob) vote for a DIFFERENT
      class than the majority of articles by raw logit? (class conflict)
  The class conflict metric is the key: disagreements should occur when the
  single most confident article overrules the crowd.
"""

import numpy as np
import matplotlib.pyplot as plt

# -- config -------------------------------------------------------------------
# 6,480 our case
N_TRIALS   = 6_480
T_ARTICLES = 10      # articles per day
N_CLASSES  = 3        # positive / negative / neutral
SEED       = 42
rng        = np.random.default_rng(SEED)


# -- helpers ------------------------------------------------------------------

def softmax(logits):
    """logits: (..., C) -> probs: (..., C)"""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def method_a(probs):
    """Average softmax probabilities -> argmax.  shape: (T, C)"""
    return int(probs.mean(axis=0).argmax())


def method_b(logits):
    """Softmax of mean logits -> argmax.  shape: (T, C)
    Note: softmax(mean(logits)).argmax() == mean(logits).argmax() always,
    since softmax is monotone -- written with softmax explicitly for clarity."""
    return int(softmax(logits.mean(axis=0)).argmax())


def spike_strength(logits):
    """Max logit minus mean of all other logits -- how much one article
    sticks out from the rest."""
    flat = logits.flatten()
    return flat.max() - np.delete(flat, flat.argmax()).mean()


def class_conflict(probs, logits):
    """
    Is there a conflict between the most confident article and the crowd?

    Most confident article = highest max-prob (reuses already-computed probs).
    Crowd = majority class by raw mean logit (= what Method B picks).

    Returns 1 if they vote for different classes, 0 if same.
    """
    confidence    = probs.max(axis=1)                  # (T,) certainty per article
    top_article   = int(confidence.argmax())           # most certain article
    top_class_a   = int(probs[top_article].argmax())   # what it votes for
    crowd_class_b = int(logits.mean(axis=0).argmax())  # what the crowd average says
    return int(top_class_a != crowd_class_b)


# -- Monte Carlo --------------------------------------------------------------
# FinBERT logits are bounded (tanh pooler -> linear), so uniform over
# [-10, 10] is more faithful than an unbounded or shifted normal.
logits_all = rng.uniform(-10, 10, size=(N_TRIALS, T_ARTICLES, N_CLASSES))

choices_a     = np.empty(N_TRIALS, dtype=int)
choices_b     = np.empty(N_TRIALS, dtype=int)
spikes        = np.empty(N_TRIALS)
conflicts     = np.empty(N_TRIALS, dtype=int)

for i in range(N_TRIALS):
    L     = logits_all[i]
    probs = softmax(L)              # compute once, reuse in method_a and class_conflict
    choices_a[i]  = method_a(probs)
    choices_b[i]  = method_b(L)
    spikes[i]     = spike_strength(L)
    conflicts[i]  = class_conflict(probs, L)

disagree_mask = choices_a != choices_b
n_disagree    = disagree_mask.sum()
agree_mask    = ~disagree_mask

print(f"Trials          : {N_TRIALS:,}")
print(f"Disagreements   : {n_disagree:,}  ({100*n_disagree/N_TRIALS:.1f}%)")
print(f"Agreements      : {N_TRIALS - n_disagree:,}  ({100*(1-n_disagree/N_TRIALS):.1f}%)")


# -- class choice tracking ----------------------------------------------------
CLASS_NAMES = ["positive", "negative", "neutral"]

print("\n-- Class choice distribution --")
print(f"{'Class':<12} {'Method A':>10} {'Method B':>10}")
for c, name in enumerate(CLASS_NAMES):
    a_pct = 100 * (choices_a == c).sum() / N_TRIALS
    b_pct = 100 * (choices_b == c).sum() / N_TRIALS
    print(f"{name:<12} {a_pct:>9.1f}% {b_pct:>9.1f}%")

print("\n-- On disagreement trials: what each method chose --")
for c, name in enumerate(CLASS_NAMES):
    a_pct = 100 * (choices_a[disagree_mask] == c).sum() / n_disagree
    b_pct = 100 * (choices_b[disagree_mask] == c).sum() / n_disagree
    print(f"{name:<12}  A chose: {a_pct:.1f}%   B chose: {b_pct:.1f}%")


# -- spike analysis -----------------------------------------------------------
spike_disagree = spikes[disagree_mask]
spike_agree    = spikes[agree_mask]

print(f"\n-- Spike strength (max logit - mean of others) --")
print(f"  Disagree trials  mean spike: {spike_disagree.mean():.3f}  median: {np.median(spike_disagree):.3f}")
print(f"  Agree    trials  mean spike: {spike_agree.mean():.3f}  median: {np.median(spike_agree):.3f}")
print(f"  -> Spike ratio (disagree/agree): {spike_disagree.mean()/spike_agree.mean():.3f}")


# -- class conflict analysis --------------------------------------------------
conflict_rate_disagree = conflicts[disagree_mask].mean()
conflict_rate_agree    = conflicts[agree_mask].mean()

print(f"\n-- Class conflict (top confident article vs crowd) --")
print(f"  Disagree trials  conflict rate: {100*conflict_rate_disagree:.1f}%")
print(f"  Agree    trials  conflict rate: {100*conflict_rate_agree:.1f}%")
print(f"  -> Conflict is {conflict_rate_disagree/conflict_rate_agree:.2f}x more common when methods disagree")
print(f"  -> Among disagree trials WITH conflict:    {100*(disagree_mask & (conflicts==1)).sum()/n_disagree:.1f}% of all disagreements")
print(f"  -> Among disagree trials WITHOUT conflict: {100*(disagree_mask & (conflicts==0)).sum()/n_disagree:.1f}% of all disagreements")


# -- figures ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Monte Carlo: Method A (avg probs) vs Method B (avg logits)\n"
    f"T={T_ARTICLES} articles, {N_TRIALS:,} trials, logits ~ N(0, 3)",
    fontsize=13,
)

# fig 1: agreement rate
ax = axes[0, 0]
ax.bar(["Agree", "Disagree"],
       [N_TRIALS - n_disagree, n_disagree],
       color=["steelblue", "tomato"], edgecolor="white")
ax.set_title("Agreement between methods")
ax.set_ylabel("# trials")
for bar, val in zip(ax.patches, [N_TRIALS - n_disagree, n_disagree]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 200,
            f"{100*val/N_TRIALS:.1f}%", ha="center", fontsize=11)

# fig 2: spike distribution -- agree vs disagree
ax = axes[0, 1]
bins = np.linspace(0, 25, 60)
ax.hist(spike_agree,    bins=bins, alpha=0.6, color="steelblue", label="Agree",    density=True)
ax.hist(spike_disagree, bins=bins, alpha=0.6, color="tomato",    label="Disagree", density=True)
ax.axvline(spike_agree.mean(),    color="steelblue", linestyle="--", linewidth=1.5,
           label=f"Agree mean={spike_agree.mean():.1f}")
ax.axvline(spike_disagree.mean(), color="tomato",    linestyle="--", linewidth=1.5,
           label=f"Disagree mean={spike_disagree.mean():.1f}")
ax.set_title("Spike strength: agree vs disagree\n(spike alone does not predict disagreement)")
ax.set_xlabel("spike strength")
ax.set_ylabel("density")
ax.legend(fontsize=9)

# fig 3: class conflict rate -- agree vs disagree
ax = axes[1, 0]
labels = ["Agree trials", "Disagree trials"]
rates  = [100 * conflict_rate_agree, 100 * conflict_rate_disagree]
bars   = ax.bar(labels, rates, color=["steelblue", "tomato"], edgecolor="white")
ax.set_title("Class conflict rate\n(top-confident article vs crowd majority)")
ax.set_ylabel("% trials with conflict")
ax.set_ylim(0, 100)
for bar, val in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 1.5, f"{val:.1f}%", ha="center", fontsize=11)

# fig 4: which class each method picks on disagreement trials
ax = axes[1, 1]
x = np.arange(N_CLASSES)
w = 0.35
a_counts = [(choices_a[disagree_mask] == c).sum() for c in range(N_CLASSES)]
b_counts = [(choices_b[disagree_mask] == c).sum() for c in range(N_CLASSES)]
ax.bar(x - w/2, a_counts, w, label="Method A (avg probs)", color="steelblue", edgecolor="white")
ax.bar(x + w/2, b_counts, w, label="Method B (avg logits)", color="orange",   edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_title("Class choices on disagreement trials")
ax.set_ylabel("# trials")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("test_log_over_porbs.png", dpi=150)
plt.show()
print("\nFigure saved -> test_log_over_porbs.png")
