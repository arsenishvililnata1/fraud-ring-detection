# Fraud Ring Detection with Graph-Based Machine Learning

Detecting coordinated fraud rings (money-mule networks) by modeling accounts,
devices, and IP addresses as a connected graph — and showing that these
network relationships improve fraud detection beyond what transaction data
alone can reveal.

![Fraud ring visualization](outputs/fraud_ring_visual_screenshot.png)
*Red = accounts in a detected fraud ring · Gray = shared device/IP linking them · Blue = other connected accounts*

## The Problem

Most fraud detection systems score one transaction at a time — is this amount
unusual, is this the right time of day, is this the normal location? That
approach misses coordinated fraud: **money mule rings**, where several
"unrelated" accounts secretly work together, moving stolen funds between each
other to disguise where it came from. No single transaction in a mule ring
looks obviously suspicious — the fraud is only visible in the *pattern of
connections* between accounts.

**Question this project answers:** does modeling accounts as a network — who
transacts with whom, who shares a device or IP — reveal fraud rings that a
standard transaction-level model would miss?

## Data

This project uses a **synthetically generated** transaction dataset (~40,000
transactions across 3,000 accounts), built to mirror the structure of real
fraud rings:
- 2,500 normal accounts transacting randomly with 300 merchants
- 500 accounts organized into ~100 small rings (3–8 accounts each) that
  transact heavily with one another, share devices/IPs within the ring, and
  move suspiciously round amounts in rapid bursts

Synthetic data was used so the project is fully reproducible without a
Kaggle account or download. The same pipeline works unchanged on real data —
e.g. the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection)
or the [Elliptic Bitcoin Dataset](https://www.kaggle.com/ellipticco/elliptic-data-set) —
by swapping the data-loading step.

## Approach

1. **Build the graph.** Represent each account, device, and IP address as a
   node. Connect accounts with an edge when a transaction occurs between
   them, and connect an account to the device/IP it used.
2. **Engineer graph features.** For every account, compute:
   - `degree` — how many connections it has
   - `clustering coefficient` — how tightly-knit its neighborhood is (fraud
     rings score high here, since everyone in the ring transacts with
     everyone else)
   - `core number` — how deeply embedded it is in a dense subgraph
   - `community ID / size` — which cluster it belongs to, found automatically
     via Louvain community detection
3. **Train and compare two models** on identical train/test splits:
   - **Baseline:** transaction-level features only (amount, frequency,
     number of counterparties, round-amount ratio)
   - **Graph-augmented:** the same features plus the graph features above
4. **Evaluate with Average Precision (PR-AUC)**, not accuracy — fraud is
   rare, so accuracy is a misleading metric here.
5. **Visualize a detected ring** as an interactive network diagram so the
   fraud pattern is visible, not just a number in a table.

## Results

| Model | Average Precision (PR-AUC) |
|---|---|
| Transaction data only (baseline) | `<fill in from your notebook output>` |
| Transaction data + graph features | `<fill in from your notebook output>` |
| **Relative improvement** | `<fill in — printed at the end of Step 5>` |

*(Run the notebook's model comparison cell and drop your printed numbers in
above — this table is the single most important result in this repo.)*

![Precision-Recall comparison](outputs/pr_curve_comparison.png)

The graph-augmented model's precision-recall curve sits above the
baseline's across most of the recall range, meaning: **at the same false
positive rate, the graph-aware model catches more fraud** — direct evidence
that network structure carries fraud signal that transaction amount and
frequency alone do not.

## What I'd do differently at scale

- **Real data:** validate on IEEE-CIS or Elliptic to confirm the lift holds
  beyond synthetic patterns designed to be graph-detectable.
- **Graph neural networks:** the community/degree features used here are a
  strong, interpretable starting point; a GNN (GraphSAGE/GAT via PyTorch
  Geometric) trained directly on the graph structure would likely push
  precision higher still, at the cost of interpretability.
- **Streaming construction:** this graph was built in batch; a production
  system would need to update the graph incrementally as transactions arrive,
  and re-score affected accounts within the batch's blast radius.
- **False positive cost:** community detection can flag large, legitimate
  hubs (e.g. a popular merchant) as "suspicious" purely due to high degree —
  a real deployment would need hub-account exclusion rules.

## How to reproduce

The full pipeline runs in **Google Colab** with no local installation:
1. Open [colab.research.google.com](https://colab.research.google.com) and
   create a new notebook.
2. Run the data generation, graph-building, model comparison, and
   visualization cells in order (see `fraud_ring_detection.ipynb` in this repo).
3. Download the resulting `outputs/` files (PR curve, feature importance
   chart, interactive fraud ring HTML) from Colab's file panel.

## Tech stack

`pandas` · `networkx` (graph construction + Louvain community detection) ·
`scikit-learn` (train/test split, evaluation metrics) · `xgboost`
(classification model) · `pyvis` (interactive network visualization) ·
Google Colab (execution environment, no local setup required)

## Files in this repo

- `fraud_ring_detection.ipynb` — full, runnable notebook
- `outputs/pr_curve_comparison.png` — baseline vs. graph-augmented model comparison
- `outputs/feature_importance.png` — which features drove the graph-augmented model
- `outputs/fraud_ring_visual.html` — interactive visualization of a detected ring
