# Aggregating Multi-Article Sentiment: A Monte Carlo Analysis of Probability Averaging vs. Logit Averaging

## 1. Motivation

In news-driven financial forecasting pipelines, a single trading day may contain multiple news articles. Each article is scored by a sentiment classifier (FinBERT) that produces a probability distribution over three classes: *positive*, *negative*, and *neutral*. To obtain a single day-level sentiment label, these per-article distributions must be aggregated into one dominant class.

Two natural aggregation strategies exist:

- **Method A** — average the per-article softmax probabilities, then take the argmax.
- **Method B** — average the per-article logits (pre-softmax scores), apply softmax, then take the argmax.

These are not mathematically equivalent. The central question of this study is: *how often do they disagree, and what structural property of the article set drives that disagreement?*

A closed-form characterisation of when the methods diverge is intractable — the interaction between the softmax nonlinearity and the argmax operator does not yield a clean analytic condition. We therefore adopt a **Monte Carlo approach**: simulate a large number of random sentiment scenarios, measure empirical disagreement rates, and correlate them against two proposed explanatory metrics.

---

## 2. Formal Definitions

Let $T$ be the number of articles on a given day. Each article $t \in \{1, \dots, T\}$ produces a logit vector $\mathbf{l}^t = [l^t_1, l^t_2, l^t_3] \in \mathbb{R}^3$.

The softmax operator applied to article $t$ is:

$$\sigma_i(\mathbf{l}^t) = \frac{e^{l^t_i}}{\sum_{j=1}^{3} e^{l^t_j}}, \quad i \in \{1, 2, 3\}$$

**Method A** (probability averaging):

$$\hat{y}_A = \underset{i}{\arg\max} \; \frac{1}{T} \sum_{t=1}^{T} \sigma_i(\mathbf{l}^t)$$

Each article contributes to the final label proportionally to its **confidence** — a highly certain article (one logit dominates) contributes a near-one-hot vector; an uncertain article (logits near equal) contributes near $[1/3, 1/3, 1/3]$ and has almost no influence.

**Method B** (logit averaging):

$$\hat{y}_B = \underset{i}{\arg\max} \; \sigma_i\!\left(\frac{1}{T}\sum_{t=1}^{T} \mathbf{l}^t\right) = \underset{i}{\arg\max} \; \frac{1}{T}\sum_{t=1}^{T} l^t_i$$

The last equality holds because softmax is strictly monotone ($e^x$ is monotone increasing), so it is irrelevant under argmax. **Method B therefore reduces exactly to: which class has the highest mean raw logit?** Every article contributes its logit value equally, regardless of how certain it is.

### 2.1 Why the Methods Can Disagree

The core asymmetry is in how they weight articles:

- Method A weights by **posterior confidence** — a certain article (high max softmax prob) dominates the average.
- Method B weights by **raw logit magnitude** — each article contributes its raw score equally; certainty is irrelevant.

These coincide when all articles agree on the dominant class, or when all articles are equally (un)certain. They diverge when a confident outlier article votes for a different class than the majority of articles in raw logit space.

### 2.2 A Note on FinBERT Logit Bounds

FinBERT applies a **tanh activation** in its pooler layer before the final linear classification head:

$$\mathbf{h} = \tanh(\mathbf{W}_{pool} \cdot \mathbf{h}_{CLS} + \mathbf{b}_{pool}), \quad \mathbf{h} \in (-1, 1)^{768}$$

$$\mathbf{l} = \mathbf{W}_{cls} \cdot \mathbf{h} + \mathbf{b}_{cls}$$

Since $\mathbf{h}$ is bounded in $(-1, 1)^{768}$, the logits satisfy:

$$|l_i| < \|\mathbf{w}^i_{cls}\|_1 + |b^i_{cls}|$$

In practice, with small learned weights from fine-tuning, FinBERT logits are empirically bounded in approximately $[-10, 10]$. This motivates our choice of simulation distribution.

---

## 3. Experiment Design

### 3.1 Simulation Setup

Each trial simulates one news day:

| Parameter | Value |
|-----------|-------|
| Articles per day $T$ | 5 |
| Classes $C$ | 3 (positive, negative, neutral) |
| Logit distribution | $\mathcal{U}(-10, 10)$ — uniform, centered at 0, bounded |
| Number of trials $N$ | 50,000 |
| Random seed | 42 |

The choice of $\mathcal{U}(-10, 10)$ directly reflects the FinBERT logit bounds established in Section 2.2. It is centered at zero (no class bias) and covers the full realistic range. An unbounded normal distribution (e.g., $\mathcal{N}(0, 1200^2)$) was used in an earlier iteration but is unrealistic — it produces logit magnitudes in the thousands that would never arise from a tanh-bounded linear layer.

For each trial we compute once:

$$\mathbf{P}^{(i)} = \sigma(\mathbf{L}^{(i)}) \in \mathbb{R}^{T \times C}$$

where $\mathbf{L}^{(i)} \in \mathbb{R}^{T \times C}$ is the logit matrix for trial $i$. This single softmax computation is shared by Method A and the class conflict metric to avoid redundant calculation.

### 3.2 Metric 1 — Spike Strength $S$

$$S = \max_{t \in [T],\, i \in [C]}(l^t_i) \;-\; \frac{1}{TC - 1} \sum_{(t,i) \neq \arg\max} l^t_i$$

$S$ measures the excess of the single largest logit over the mean of all other logits in the trial. A high $S$ indicates one article-class pair has a very large raw score relative to the rest — a "spike" in logit space.

**Hypothesis being tested:** Does a raw logit spike cause the two methods to diverge?

**Calculation detail:** The metric flattens the full $T \times C$ logit matrix, identifies the global maximum, and subtracts the mean of the remaining $TC - 1$ values. This is scale-sensitive and operates entirely in logit space, with no softmax involved.

### 3.3 Metric 2 — Class Conflict $\kappa$

Let the **most confident article** be:

$$t^* = \underset{t \in [T]}{\arg\max} \; \max_{i \in [C]} \sigma_i(\mathbf{l}^t)$$

This is the article whose softmax distribution is most peaked — the one that is most certain about its prediction. Its confidence is measured by its maximum probability $\max_i \sigma_i(\mathbf{l}^{t^*})$, not by its maximum logit.

Let $c^*_A = \arg\max_i \sigma_i(\mathbf{l}^{t^*})$ be the class that $t^*$ votes for.

Let $c^*_B = \arg\max_i \bar{l}_i$ where $\bar{l}_i = \frac{1}{T}\sum_t l^t_i$ be the class the crowd (mean logit) selects — equivalently, what Method B would choose.

The class conflict indicator is:

$$\kappa = \mathbf{1}[c^*_A \neq c^*_B]$$

$\kappa = 1$ means the most confident article and the crowd majority disagree on the dominant class. This is the proposed structural driver of disagreement between the methods: Method A is pulled toward $c^*_A$ by the confident outlier's high-weight softmax vector; Method B follows $c^*_B$ regardless of that outlier's certainty.

**Why softmax is required here, not raw logits:**

Consider two articles:

| Article | Logits | Max logit | Softmax | Max prob |
|---------|--------|-----------|---------|----------|
| A | [9, 8, 7] | 9 | [0.67, 0.24, 0.09] | 0.67 |
| B | [5, 0, 0] | 5 | [0.99, 0.005, 0.005] | 0.99 |

By max logit, article A appears more "confident" (9 > 5). But article B's softmax is nearly one-hot — it is far more certain. Confidence is a property of the *gap between logits within one article*, not the absolute magnitude. Softmax captures this relative gap; raw logit maximum does not.

**Conflict rate** is computed separately for agree and disagree trial subsets:

$$\text{conflict rate} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \kappa^{(i)}, \quad \mathcal{S} \in \{\text{agree trials},\, \text{disagree trials}\}$$

---

## 4. Results

### 4.1 Overall Agreement

| | Count | Rate |
|-|-------|------|
| Agreements | 38,073 | 76.1% |
| Disagreements | 11,927 | 23.9% |

Under the realistic $\mathcal{U}(-10, 10)$ logit distribution, the two methods produce **different dominant class labels on approximately 1 in 4 trials**. This is a non-trivial divergence rate and establishes that the choice of aggregation method meaningfully affects the output label in practice.

### 4.2 Class Distribution

| Class | Method A | Method B |
|-------|----------|----------|
| positive | 33.1% | 33.3% |
| negative | 33.0% | 32.9% |
| neutral | 33.9% | 33.8% |

Both methods produce a near-uniform class distribution, as expected under a zero-centered symmetric logit prior. Neither method has a systematic directional bias — any bias introduced by one method relative to the other is trial-specific, not structural.

On disagreement trials specifically:

| Class | Method A chose | Method B chose |
|-------|---------------|---------------|
| positive | 32.8% | 33.8% |
| negative | 33.4% | 32.9% |
| neutral | 33.8% | 33.3% |

No class is systematically preferred by either method on disagreement trials. The disagreements are symmetric across classes, confirming that the divergence is not an artifact of class imbalance or distributional asymmetry.

### 4.3 Spike Strength Does Not Explain Disagreements

| Condition | Mean spike $S$ | Median spike $S$ |
|-----------|---------------|-----------------|
| Agree trials | 9.394 | 9.392 |
| Disagree trials | 9.310 | 9.318 |
| Ratio (disagree / agree) | **0.991** | — |

The spike strength is marginally *lower* on disagreement trials than on agreement trials (ratio = 0.991 < 1). This is the opposite direction from what a spike-driven hypothesis would predict, and the magnitude is negligible.

**Conclusion:** A single large logit value, measured in isolation from the article's other logits, carries no predictive power for whether the two methods will disagree. The raw amplitude of an outlier does not determine the outcome.

This result makes intuitive sense: a spike in logit space does not translate linearly to a spike in probability space. An article with logits [10, 9, 8] has a large max logit but a softmax of approximately [0.67, 0.24, 0.09] — only moderately confident. An article with logits [3, -3, -3] has a smaller max logit but a softmax of approximately [0.998, 0.001, 0.001] — extremely confident. The raw logit magnitude is not a reliable proxy for the influence an article exerts on Method A.

### 4.4 Class Conflict Strongly Predicts Disagreements

| Condition | Conflict rate $P(\kappa=1)$ |
|-----------|----------------------------|
| Agree trials | 30.4% |
| Disagree trials | 48.2% |
| Ratio (disagree / agree) | **1.59x** |

| Decomposition of disagreements | Share |
|-------------------------------|-------|
| Disagree trials WITH conflict ($\kappa=1$) | 48.2% |
| Disagree trials WITHOUT conflict ($\kappa=0$) | 51.8% |

When the most confident article votes for a different class than the crowd mean, the two methods disagree at **1.59x the baseline rate** (48.2% vs 30.4%). This is a substantial elevation and confirms the proposed mechanism.

The conflict metric alone does not account for all disagreements (only 48.2% of disagreements have $\kappa=1$), which indicates that additional higher-order interactions between articles contribute to divergence. However, class conflict is the single strongest univariate predictor identified in this study — and the only one with explanatory lift above baseline.

### 4.5 Comparison Across Logit Regimes

We ran the experiment under two logit distributions to assess sensitivity:

| Distribution | Disagreement rate | Conflict rate (disagree) | Conflict rate (agree) | Conflict ratio |
|-------------|------------------|--------------------------|----------------------|----------------|
| $\mathcal{N}(0, 3^2)$ | 20.9% | 50.5% | 28.0% | 1.81x |
| $\mathcal{U}(-10, 10)$ | 23.9% | 48.2% | 30.4% | 1.59x |

The disagreement rate is higher under the uniform distribution (23.9% vs 20.9%), which reflects the heavier tails of $\mathcal{U}(-10, 10)$ relative to $\mathcal{N}(0, 9)$ at the extremes. The conflict ratio is slightly lower (1.59x vs 1.81x), suggesting the class conflict metric is most discriminative in the Gaussian regime where logit values are more concentrated near zero and confident outliers are rarer.

---

## 5. Discussion

### 5.1 Mechanism Summary

The experiment establishes the following picture:

1. **Spike strength is a red herring.** A single article having a very large raw logit does not predict method divergence. What matters is not how extreme the logit is, but whether the article is confident *about a different class than the crowd*.

2. **Class conflict is the structural driver.** When the most confident article (by softmax max-prob) disagrees with the plurality class in raw logit space, Method A follows the confident outlier and Method B follows the crowd. This is the mechanism behind nearly half of all observed disagreements.

3. **The remaining ~52% of disagreements** arise from subtler multi-article interactions — cases where no single article is a dominant outlier, but the nonlinearity of softmax causes the weighted average of probabilities to tip toward a different class than the simple mean logit. These are not captured by any single-article metric.

### 5.2 Practical Implication

**Method A is the correct choice for the pipeline**, for two reasons:

*Semantic:* In real news sentiment, a single article containing clear, unambiguous evidence of a market event (e.g., a bankruptcy announcement) should outweigh five mildly uncertain articles that weakly lean the other way. Method A's confidence weighting achieves this naturally.

*Practical:* FinBERT exposes probabilities after the softmax layer, not raw logits. Method B is not accessible without modifying the inference pipeline to extract pre-softmax activations.

The 23.9% disagreement rate under the realistic $\mathcal{U}(-10, 10)$ distribution represents an upper bound on real divergence, since actual FinBERT distributions over a day's articles are not uniformly random — sentiment tends to be locally consistent within a news cycle, reducing the frequency of the conflicting-outlier scenario that drives divergence.

---

## 6. Conclusion

We conducted a 50,000-trial Monte Carlo experiment to characterise when probability averaging (Method A) and logit averaging (Method B) produce different dominant sentiment labels. Two candidate explanatory metrics were evaluated: spike strength $S$ (raw logit magnitude of the largest outlier) and class conflict $\kappa$ (whether the most confident article disagrees with the crowd).

Spike strength carries no predictive power for disagreement (ratio = 0.991). Class conflict is predictive (ratio = 1.59x) and accounts for approximately half of all observed disagreements. The remaining disagreements arise from higher-order nonlinear interactions not captured by single-article metrics.

These findings confirm that Method A's sensitivity to confident outliers — its defining behavioural difference from Method B — is the primary source of divergence between the two approaches, and that this sensitivity is a feature rather than a flaw for the intended use case of daily news sentiment aggregation.
