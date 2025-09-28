# Model Versions Explained

Here’s what the three versions mean, in simple terms, with quick project examples.

---

## v1 — Base model

- Smallest, simplest version. Sets a baseline.
- Fast to train, lowest energy/emissions, lowest risk of overfitting.

**Examples:**

- **DLinear v1**: two `Dense(72)` heads (seasonal + trend) → Add (~10.5K params).
- **Transformer v1**: light attention stack, tiny feed-forward, no/limited positional info.
- **Robust Hybrid v1**: `Conv1D` → `MHA×2` → `BiLSTM×2` + a small linear gate.

---

## v2 — Modest capacity/regularization changes

- Small, targeted upgrades to improve stability and generalization.
- Typical tweaks: a few more units/filters, add Dropout/L2/LayerNorm, add positional embeddings, mild head changes.
- Slightly slower, slightly higher emissions, usually better accuracy.

**Examples:**

- **DLinear v2**: `MovingAverage` to isolate trend + same small heads (~10.5K params), adds regularization idea.
- **Transformer v2**: adds positional + token embedding, wider embedding (e.g., 256), fewer but stronger blocks.
- **Robust Hybrid v2**: depthwise separable Conv blocks, squeeze-excite, BiLSTM+GRU fusion, stronger gating + Dropout.

---

## v3 — Scales depth/width

- Bigger jump in model size: more layers (deeper) and/or many more units (wider).
- Highest accuracy potential; costs more compute time and energy.
- Needs regularization to stay stable (Dropout, L2, Pre-LN).

**Examples:**

- **DLinear v3**: two 512-unit MLP branches + Dropout → `Dense(72)` per branch, then Add (~148K params).
- **Transformer v3**: 4 attention blocks, more heads, L2/Huber loss, deeper output head.
- **Robust Hybrid v3** (if present): would further stack attention/recurrent paths or widen channels.

---

## Rule of thumb

- Start with **v1** to validate the pipeline.
- Move to **v2** for a balanced “good accuracy, still efficient.”
- Use **v3** when you need the best accuracy and can afford more compute/emissions.
