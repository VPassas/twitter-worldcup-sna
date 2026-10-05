# Twitter World Cup — Social Network Analysis & Recommender

> Large-scale social-network analysis of ~5.1M tweets from the **2014 FIFA World Cup**:
> influence ranking, community detection, and a machine-learning link-prediction
> recommender — built end-to-end on a **2M-node / 3.9M-edge** interaction graph.

**Tech stack:** Python · NetworkX · scikit-learn · NumPy/SciPy · Matplotlib · TF-IDF + Truncated SVD · Louvain · MLP / kNN

### Highlights
- Built a directed user-to-user graph (mention / reply / retweet) from 5.1M raw tweets → **2,044,100 nodes, 3,929,041 edges**.
- Tested the power-law hypothesis on in/out-degree distributions (log-log OLS fit).
- Ranked influencers by degree centrality, **PageRank**, and exact **betweenness**, then characterised them with TF-IDF topic tags.
- Detected **~41k communities** with Louvain (modularity ≈ 0.75) and labelled them by dominant hashtags/topics.
- Engineered hybrid user embeddings (network + content features) and evaluated a link-prediction recommender with **Precision@k / Recall@k**.

> Course project — Deree, The American College of Greece · ITC 4441 Web Science & Social Media Platform Analytics · Spring 2026
> Author: Evangelos Passas

## Dataset

Tweets from the *Twitter World Cup* collection (`tweets.json.*` line-delimited JSON).
The raw dataset is **large (several GB)** and is therefore **not** committed to this
repository. Place the extracted tweet files in a folder named `twitter_world_cup_part1/`
next to the scripts:

```
final/
├── twitter_world_cup_part1/     # tweets.json.0, tweets.json.1, ...  (not in git)
├── project_output/              # generated results (created automatically)
├── q1_influence.py
├── q2_communities.py
└── q3_recommender.py
```

A user-to-user **directed** edge `A → B` is created whenever user A mentions, replies
to, or retweets user B. Loaded totals for the full dataset: ~5.1M tweets, ~2.4M users,
with a resulting network of **2,044,100 nodes / 3,929,041 edges**.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

The scripts must be run **in order** — Q2 and Q3 read the network, centrality, and
community files produced by earlier steps.

```bash
python q1_influence.py      # builds network, centralities, degree distributions
python q2_communities.py    # Louvain communities + per-community topics
python q3_recommender.py    # user vectors, link-prediction model, Precision@k/Recall@k
```

On Windows you can run the whole pipeline with `run_all.bat`.

All outputs are written to `project_output/`.

## What each script does

### Q1 — Influence (`q1_influence.py`)
- Builds the directed user-to-user network and saves it to `network-Passas2026.data`.
- Investigates the **power-law** hypothesis via log-log in/out-degree distributions with
  an OLS fit (in-degree slope ≈ −0.85, out-degree slope ≈ −2.42) → `degree_distributions.png`.
- Ranks users by **in/out-degree centrality**, **PageRank**, and **betweenness** (exact,
  on the top in-degree subgraph), saving `degree.data`, `pageRank.data`, `betweeness.data`.
- Prints the **top-10 influential users** per metric, tags them with TF-IDF keywords from
  their tweets, and compares the three rankings (overlap + whether betweenness-central
  users bridge themes).

### Q2 — Communities (`q2_communities.py`)
- Runs **Louvain** community detection on the undirected network
  (~41,455 communities, modularity ≈ 0.745) → `communities2026.data`
  (`user_id, screen_name, community_id`).
- Extracts dominant **hashtags / keywords** per community with TF-IDF → `community_sizes.png`.
- Classifies each large community as an *interest group*, *discussion cluster*, or
  *information-sharing network* using internal density and hub ratio →
  `community_classification.png`.

### Q3 — Recommender (`q3_recommender.py`)
- Represents each user as a **hybrid feature vector** — network features
  (in/out-degree centrality, PageRank, clustering coefficient, community id) + content
  features (TF-IDF → TruncatedSVD topic embedding) → `user_vectors2026.data`.
- Frames recommendation as **link prediction**: trains an MLP and a kNN classifier on
  connected/non-connected node pairs, comparing **Structural**, **Text**, and **Hybrid**
  feature sets.
- Evaluates with **Precision@k / Recall@k** (k = 5, 10, 20) → `precision_recall_at_k.png`.

Precision@k on the held-out split:

| Method     | P@5    | P@10   | P@20   |
|------------|--------|--------|--------|
| Structural | 0.0376 | 0.0317 | 0.0263 |
| Text       | 0.0115 | 0.0114 | 0.0110 |
| Hybrid     | 0.0299 | 0.0276 | 0.0224 |

## Outputs

| File | Produced by | Description |
|------|-------------|-------------|
| `network-Passas2026.data` | Q1 | Directed weighted edge list |
| `degree.data`, `pageRank.data`, `betweeness.data` | Q1 | Centrality rankings |
| `degree_distributions.png` | Q1 | In/out-degree log-log plots |
| `communities2026.data` | Q2 | `user_id, screen_name, community_id` |
| `community_sizes.png`, `community_classification.png` | Q2 | Community plots |
| `user_vectors2026.data` | Q3 | Hybrid user feature vectors |
| `precision_recall_at_k.png` | Q3 | Recommender evaluation |

Large `.data` outputs are git-ignored (regenerate by running the scripts); the `.png`
plots are kept in the repo so results are visible without the dataset.

## Assignment mapping

| Requirement | Where |
|-------------|-------|
| Q1 Influence (35%) — network, power-law, centralities, top-10 | `q1_influence.py` |
| Q2 Communities (30%) — detection, topics, interpretation | `q2_communities.py` |
| Q3 Recommender (35%) — user vectors, ML model, Precision@k/Recall@k | `q3_recommender.py` |

The written report is submitted separately (Turnitin) and is not included here.
