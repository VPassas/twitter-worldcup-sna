import json
import os
import csv
import numpy as np
import networkx as nx
from networkx.algorithms.community import modularity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'twitter_world_cup_part1')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'project_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load network from Q1 output
print("Loading network from Q1 output...")
netfile = os.path.join(OUTPUT_DIR, 'network-Passas2026.data')
G = nx.read_edgelist(netfile, create_using=nx.DiGraph(), data=[('weight', float)])
G_und = G.to_undirected()
print('#nodes=', G_und.number_of_nodes())
print('#edges=', G_und.number_of_edges())

#stream tweets for TF-IDF community analysis
print("Loading tweet content for community analysis...")

files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('tweets.json.')],
               key=lambda x: int(x.split('.')[-1]))

user_tweets   = defaultdict(list)
user_hashtags = defaultdict(list)
user_names    = {}

for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    print("  reading", fname)
    infile = open(fpath, 'r', encoding='utf-8')
    c = infile.readline()
    while c != '':
        line = c.strip()
        if line:
            try:
                tw = json.loads(line)
                uid   = str(tw['user']['id'])
                sname = tw['user']['screen_name']
                ttext = tw.get('text', '')
                user_names[uid] = sname
                user_tweets[uid].append(ttext)
                for ht in tw.get('entities', {}).get('hashtags', []):
                    user_hashtags[uid].append(ht['text'].lower())
                for m in tw.get('entities', {}).get('user_mentions', []):
                    tid = str(m['id'])
                    user_names[tid] = m.get('screen_name', '')
            except:
                pass
        c = infile.readline()
    infile.close()

print("Tweet content loaded for %d users" % len(user_tweets))


#Q2
print("\nQ2 - Communities of Users")

print("Applying Louvain community detection...")
commLouvain = nx.community.louvain_communities(G_und, seed=42)
nComm = len(commLouvain)

mod = modularity(G_und, commLouvain)
print("Louvain number of communities:", nComm)
print("Modularity Louvain:", round(mod, 4))

commLouvain_sorted = sorted(commLouvain, key=len, reverse=True)
condTop10 = [nx.conductance(G_und, c, weight='weight') for c in commLouvain_sorted[:10]]
print("Mean conductance (top 10):", round(np.mean(condTop10), 4))

partition = {}
for cid, members in enumerate(commLouvain):
    for uid in members:
        partition[uid] = cid

commFile = os.path.join(OUTPUT_DIR, 'communities2026.data')
with open(commFile, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'screen_name', 'community_id'])
    for uid, cid in sorted(partition.items(), key=lambda x: x[1]):
        w.writerow([uid, user_names.get(uid, ''), cid])
print("Saved communities2026.data")

commSizes = Counter(partition.values())
topComms  = commSizes.most_common(10)
print("Top 10 communities by size:")
for cid, sz in topComms:
    print("  Community %d: %d users" % (cid, sz))

hg_comm = [sz for _, sz in commSizes.most_common(20)]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sorted(hg_comm, reverse=True), '.')
ax.set_xlabel('Community Rank')
ax.set_ylabel('Number of Users')
ax.set_title('Community Sizes (Louvain)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'community_sizes.png'), dpi=150)
plt.close()
print("Saved community_sizes.png")


#dominant hashtags/keywords per community using TF-IDF
print("Dominant hashtags/keywords per community (TF-IDF):")

commUsers  = defaultdict(list)
for uid, cid in partition.items():
    commUsers[cid].append(uid)

topCommIds = [cid for cid, _ in topComms]
commData   = {}

my_stop_words = text.ENGLISH_STOP_WORDS.union([
    'http', 'https', 'amp', 'com', 'www', 'pic', 'twitter',
    'tweet', 'retweet', 'via', 'rt', 'co'])

commCorpus = [' '.join([' '.join(user_tweets.get(uid, [])) for uid in commUsers[cid]])
              for cid in topCommIds]

vectorizer  = TfidfVectorizer(stop_words=list(my_stop_words))
vectorizer.fit(commCorpus)
features    = np.array(vectorizer.get_feature_names_out())
corpus_vec  = vectorizer.transform(commCorpus).toarray()

for i, cid in enumerate(topCommIds):
    members = commUsers[cid]
    ht_all  = []
    for uid in members:
        ht_all.extend(user_hashtags.get(uid, []))
    indexes = corpus_vec[i, :].argsort()[-10:][::-1]
    topKw   = list(features[indexes])
    topHt   = Counter(ht_all).most_common(10)
    commData[cid] = {'size': len(members), 'hashtags': topHt, 'keywords': topKw}
    print("  Community %d (%d users):" % (cid, len(members)))
    print("    Top hashtags:", ['#' + t + '(' + str(c) + ')' for t, c in topHt[:7]])
    print("    Top keywords:", topKw[:7])


# community classification
print("\nCommunity interpretation:")

densities = []
hubRatios = []
ctypes    = []

for cid in topCommIds:
    info    = commData[cid]
    members = commUsers[cid]
    kwNames = info['keywords'][:5]

    sub        = G_und.subgraph(members)
    nEdgesInternal = sub.number_of_edges()
    nPossible  = info['size'] * (info['size'] - 1) / 2 if info['size'] > 1 else 1
    density    = nEdgesInternal / nPossible

    degs     = dict(sub.degree())
    maxDeg   = max(degs.values()) if degs else 0
    hubRatio = maxDeg / max(info['size'] - 1, 1)

    if hubRatio > 0.3:
        ctype = "information-sharing network"
    elif density > 0.001:
        ctype = "interest group"
    else:
        ctype = "discussion cluster"

    densities.append(density)
    hubRatios.append(hubRatio)
    ctypes.append(ctype)

    print("  Community %d (%d users, density=%.6f, hubRatio=%.3f): %s" % (
        cid, info['size'], density, hubRatio, ctype))
    print("    keywords:", kwNames)
    print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
xPos = range(len(topCommIds))
axes[0].bar(xPos, densities, color='steelblue', alpha=0.8)
axes[0].set_xticks(xPos)
axes[0].set_xticklabels(['C%d' % c for c in topCommIds], fontsize=7, rotation=45)
axes[0].set_ylabel('internal density')
axes[0].set_title('Community internal density')
axes[0].grid(axis='y', alpha=0.3)
axes[1].bar(xPos, hubRatios, color='tomato', alpha=0.8)
axes[1].set_xticks(xPos)
axes[1].set_xticklabels(['C%d' % c for c in topCommIds], fontsize=7, rotation=45)
axes[1].set_ylabel('hub ratio')
axes[1].set_title('Community hub ratio')
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'community_classification.png'), dpi=150)
plt.close()
print("Saved community_classification.png")


print("\nDone. Q2 output files in:", OUTPUT_DIR)
for f in ['communities2026.data', 'community_sizes.png', 'community_classification.png']:
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(fp):
        print("  %-40s %10d bytes" % (f, os.path.getsize(fp)))
