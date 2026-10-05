import json
import os
import csv
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import random

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'twitter_world_cup_part1')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'project_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load network from Q1 output
print("Loading network from Q1 output...")
netfile = os.path.join(OUTPUT_DIR, 'network-Passas2026.data')
G_dir = nx.read_edgelist(netfile, create_using=nx.DiGraph(), data=[('weight', float)])
G_und = G_dir.to_undirected()
print('#nodes=', G_und.number_of_nodes())
print('#edges=', G_und.number_of_edges())

#load centrality values from Q1 output
print("Loading centrality values from Q1 output...")

in_cen  = {}
out_cen = {}
with open(os.path.join(OUTPUT_DIR, 'degree.data'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        in_cen[row['user_id']]  = float(row['in_degree_centrality'])
        out_cen[row['user_id']] = float(row['out_degree_centrality'])

pgNodes = {}
with open(os.path.join(OUTPUT_DIR, 'pageRank.data'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pgNodes[row['user_id']] = float(row['pagerank'])

print("Centrality loaded for %d users" % len(in_cen))

#load community assignments from Q2 output
print("Loading community assignments from Q2 output...")
partition = {}
with open(os.path.join(OUTPUT_DIR, 'communities2026.data'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        partition[row['user_id']] = int(row['community_id'])
print("Communities loaded for %d users" % len(partition))

#stream tweets for TF-IDF content features
print("Loading tweet content for user vectors...")

files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('tweets.json.')],
               key=lambda x: int(x.split('.')[-1]))

user_tweets = defaultdict(list)
user_names  = {}

for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    print("  reading", fname)
    infile = open(fpath, 'r', encoding='utf-8')
    c = infile.readline()
    while c != '':
        line = c.strip()
        if line:
            try:
                tw    = json.loads(line)
                uid   = str(tw['user']['id'])
                sname = tw['user']['screen_name']
                ttext = tw.get('text', '')
                user_names[uid] = sname
                user_tweets[uid].append(ttext)
                for m in tw.get('entities', {}).get('user_mentions', []):
                    tid = str(m['id'])
                    user_names[tid] = m.get('screen_name', '')
            except:
                pass
        c = infile.readline()
    infile.close()

print("Tweet content loaded for %d users" % len(user_tweets))


#Q3
print("\nQ3 - Recommender System")
print("\nLink prediction using 3 methods: Structural, Text, Hybrid")

#clustering coefficient
print("Computing clustering coefficient...")
clust = nx.clustering(G_und)

# TF-IDF + SVD content features
print("Computing content features (TF-IDF + SVD)...")
N_CONTENT = 10
my_stop_words_q3 = text.ENGLISH_STOP_WORDS.union([
    'http', 'https', 'amp', 'rt', 'co', 'via',
    'worldcup2014', 'worldcup', 'world', 'cup', 'brazil', '2014'])

contentUids = [uid for uid in user_tweets if uid in G_und]
contentDocs = [' '.join(user_tweets[uid]) for uid in contentUids]

contentVec    = TfidfVectorizer(stop_words=list(my_stop_words_q3), max_features=5000, min_df=2)
tfidfMatrix   = contentVec.fit_transform(contentDocs)
svd           = TruncatedSVD(n_components=N_CONTENT, random_state=42)
contentLatent = normalize(svd.fit_transform(tfidfMatrix))
contentByUid  = {uid: contentLatent[i] for i, uid in enumerate(contentUids)}
print("Content features shape:", contentLatent.shape)

#assemble hybrid vector: [in_cen, out_cen, pagerank, clustering, comm_norm, topic_0..9]
maxComm = max(partition.values()) if partition else 1
userVectors = {}
for uid in G_und.nodes():
    netFeats  = [
        in_cen.get(uid, 0.0),
        out_cen.get(uid, 0.0),
        pgNodes.get(uid, 0.0),
        clust.get(uid, 0.0),
        partition.get(uid, 0) / max(maxComm, 1)
    ]
    contFeats = list(contentByUid[uid]) if uid in contentByUid else [0.0] * N_CONTENT
    userVectors[uid] = np.array(netFeats + contFeats, dtype=float)

print("User vectors built for %d users" % len(userVectors))

# Build structural-only vectors (graph features only, 5-dim)
structVectors = {}
for uid in G_und.nodes():
    structVectors[uid] = np.array([
        in_cen.get(uid, 0.0),
        out_cen.get(uid, 0.0),
        pgNodes.get(uid, 0.0),
        clust.get(uid, 0.0),
        partition.get(uid, 0) / max(maxComm, 1)
    ], dtype=float)

# Build text-only vectors (TF-IDF/SVD features only, 10-dim)
textVectors = {}
for uid in G_und.nodes():
    textVectors[uid] = np.array(
        list(contentByUid[uid]) if uid in contentByUid else [0.0] * N_CONTENT,
        dtype=float
    )
print("Structural and text vectors built.")

# save user_vectors2026.data (hybrid)
vecFile   = os.path.join(OUTPUT_DIR, 'user_vectors2026.data')
featNames = ['in_deg_cen', 'out_deg_cen', 'pagerank', 'clustering', 'comm_norm'] + \
            ['topic_%d' % i for i in range(N_CONTENT)]
with open(vecFile, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'screen_name'] + featNames)
    for uid, vec in userVectors.items():
        w.writerow([uid, user_names.get(uid, '')] + list(vec))
print("Saved user_vectors2026.data (%d users, %d dims)" % (len(userVectors), 5 + N_CONTENT))

# subgraph: top nodes by degree for tractable ML training
MAX_NODES  = 5000
degSorted  = sorted(G_und.degree(), key=lambda x: x[1], reverse=True)
activeNodes = [n for n, _ in degSorted[:MAX_NODES * 3]
               if G_und.degree(n) >= 3 and n in userVectors][:MAX_NODES]

graph = G_und.subgraph(activeNodes).copy()
graph.remove_nodes_from(list(nx.isolates(graph)))
print('Number of Nodes=', len(graph.nodes))
print('Number of Edges=', len(graph.edges))

# split edges into train and test
def split_graph_edges(G, test_ratio=0.2, seed=42):
    rng   = random.Random(seed)
    edges = list(G.edges())
    rng.shuffle(edges)
    n_test     = int(len(edges) * test_ratio)
    test_edges  = edges[:n_test]
    train_edges = edges[n_test:]
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    G_test = nx.Graph()
    G_test.add_nodes_from(G.nodes())
    G_test.add_edges_from(test_edges)
    return G_train, G_test, train_edges, test_edges

g_train, g_test, train_edges, test_edges = split_graph_edges(graph, test_ratio=0.5)
print("Original:", graph.number_of_edges())
print("Train:",    g_train.number_of_edges())
print("Test:",     g_test.number_of_edges())

nodes_connectedTr = list(g_train.edges())
nodes_connectedTe = list(g_test.edges())
nodeListTr = list(g_train.nodes())
nodeListTe = list(g_test.nodes())

sampleSizeTr = len(nodes_connectedTr)
sampleSizeTe = len(nodes_connectedTe)

random.seed(42)
nodes_notConnectedTr = []
while len(nodes_notConnectedTr) < sampleSizeTr:
    u, v = random.sample(nodeListTr, 2)
    if not g_train.has_edge(u, v):
        nodes_notConnectedTr.append((u, v))

nodes_notConnectedTe = []
while len(nodes_notConnectedTe) < sampleSizeTe:
    u, v = random.sample(nodeListTe, 2)
    if not g_test.has_edge(u, v):
        nodes_notConnectedTe.append((u, v))

print('Number of pairs of not connected Nodes (train)=', len(nodes_notConnectedTr))
print('Number of pairs of connected Nodes (train)=',    len(nodes_connectedTr))

# testByUser is the same for all methods
testByUser = defaultdict(set)
for u, v in test_edges:
    testByUser[u].add(v)
    testByUser[v].add(u)

recNodes = list(graph.nodes())
K_VALUES = [5, 10, 20]

# Method definitions: structural (5-dim), text (10-dim), hybrid (15-dim)
METHODS = [
    ('Structural', structVectors, 5),
    ('Text',       textVectors,   N_CONTENT),
    ('Hybrid',     userVectors,   5 + N_CONTENT),
]

def build_pair_matrix(pairs, vecs, ndims):
    mat = np.zeros((len(pairs), 2 * ndims))
    for i, (u, v) in enumerate(pairs):
        mat[i, :] = np.concatenate((vecs[u], vecs[v]))
    return mat

results_summary = {}

for method_name, vecs, ndims in METHODS:
    print('\n' + '='*60)
    print('METHOD: %s (%d-dim vectors)' % (method_name, ndims))
    print('='*60)

    data1Tr = build_pair_matrix(nodes_connectedTr,    vecs, ndims)
    data2Tr = build_pair_matrix(nodes_notConnectedTr, vecs, ndims)
    data1Te = build_pair_matrix(nodes_connectedTe,    vecs, ndims)
    data2Te = build_pair_matrix(nodes_notConnectedTe, vecs, ndims)

    print(data1Tr.shape)
    print(data2Tr.shape)

    allDataTr = np.concatenate((data1Tr, data2Tr), axis=0)
    allDataTe = np.concatenate((data1Te, data2Te), axis=0)
    yTr = np.concatenate((np.ones(len(data1Tr)), np.zeros(len(data2Tr))))
    yTe = np.concatenate((np.ones(len(data1Te)), np.zeros(len(data2Te))))

    x_train, y_train = allDataTr, yTr
    x_test,  y_test  = allDataTe, yTe

    print('Train matrix size', x_train.shape)
    print('Test matrix size',  x_test.shape)

    clfANN = MLPClassifier(solver='adam', activation='relu',
                           batch_size=10, tol=1e-7, validation_fraction=0.2,
                           hidden_layer_sizes=(15, 10), random_state=1,
                           max_iter=50000, verbose=False)
    clfkNN = KNeighborsClassifier(n_neighbors=3)

    clfANN.fit(x_train, y_train)
    clfkNN.fit(x_train, y_train)

    y_train_pred_ANN = clfANN.predict(x_train)
    y_train_pred_kNN = clfkNN.predict(x_train)
    y_test_pred_ANN  = clfANN.predict(x_test)
    y_test_pred_kNN  = clfkNN.predict(x_test)

    print('\nConf matrix, Train Set, Neural Net')
    print(confusion_matrix(y_train, y_train_pred_ANN))
    print()
    print('Conf matrix, Test Set, Neural Net')
    print(confusion_matrix(y_test, y_test_pred_ANN))
    print()
    print('\nConf matrix, Train Set, Nearest Neighbour')
    print(confusion_matrix(y_train, y_train_pred_kNN))
    print()
    print('Conf matrix, Test Set, Nearest Neighbour')
    print(confusion_matrix(y_test, y_test_pred_kNN))
    print()

    print('\nNeural Networks')
    print(' 1: link exists,   0: no link')
    print(classification_report(y_test, y_test_pred_ANN))

    print('\nNearest Neighbour')
    print(' 1: link exists,   0: no link')
    print(classification_report(y_test, y_test_pred_kNN))

    # Precision@k and Recall@k
    print("\nEvaluating Precision@k, Recall@k (%s)..." % method_name)
    evalUsers = [u for u in testByUser if u in vecs]
    prec_at_k = {k: [] for k in K_VALUES}
    rec_at_k  = {k: [] for k in K_VALUES}

    printed = 0
    for u in evalUsers:
        trueConn  = testByUser[u]
        trainNbrs = set(g_train.neighbors(u)) if u in g_train else set()
        cands = [v for v in recNodes if v != u and v not in trainNbrs and v in vecs]
        if not cands:
            continue

        vecU   = vecs[u]
        X_cand = np.array([np.concatenate((vecU, vecs[v])) for v in cands])
        probs  = clfANN.predict_proba(X_cand)[:, 1]
        ranked_idx = np.argsort(-probs)
        ranked = [cands[i] for i in ranked_idx]

        if printed < 2:
            uname = user_names.get(u, u)
            print("  Recommendations for @%s:" % uname)
            for ri in range(min(5, len(ranked))):
                v   = ranked[ri]
                p   = probs[ranked_idx[ri]]
                hit = " << HIT" if v in trueConn else ""
                print("    %d. @%-20s P=%.4f%s" % (ri + 1, user_names.get(v, v), p, hit))
            printed += 1

        for k in K_VALUES:
            topk = ranked[:k]
            hits = sum(1 for v in topk if v in trueConn)
            nRel = len(trueConn) if trueConn else 1
            prec_at_k[k].append(hits / k)
            rec_at_k[k].append(hits / nRel)

    print("Precision@k and Recall@k (%s, %d users):" % (method_name, len(evalUsers)))
    print("  k     Precision@k    Recall@k")
    print("  ---   -----------    --------")
    for k in K_VALUES:
        p = np.mean(prec_at_k[k]) if prec_at_k[k] else 0
        r = np.mean(rec_at_k[k])  if rec_at_k[k]  else 0
        print("  %-5d  %.4f         %.4f" % (k, p, r))

    results_summary[method_name] = {
        'prec': {k: np.mean(prec_at_k[k]) if prec_at_k[k] else 0 for k in K_VALUES},
        'rec':  {k: np.mean(rec_at_k[k])  if rec_at_k[k]  else 0 for k in K_VALUES},
        'clf':  clfANN
    }

# Summary comparison across all 3 methods
print('\n' + '='*60)
print('COMPARISON: Precision@k across all methods')
print('%-12s  %8s  %8s  %8s' % ('Method', 'P@5', 'P@10', 'P@20'))
for mn, res in results_summary.items():
    print('%-12s  %8.4f  %8.4f  %8.4f' % (mn, res['prec'][5], res['prec'][10], res['prec'][20]))

# Plot: all 3 methods on same axes
colors  = {'Structural': 'green', 'Text': 'blue', 'Hybrid': 'red'}
markers = {'Structural': 's-',    'Text': 'o-',   'Hybrid': '^-'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for mn, res in results_summary.items():
    axes[0].plot(K_VALUES, [res['prec'][k] for k in K_VALUES],
                 markers[mn], color=colors[mn], label=mn)
    axes[1].plot(K_VALUES, [res['rec'][k]  for k in K_VALUES],
                 markers[mn], color=colors[mn], label=mn)
axes[0].set_xlabel('k')
axes[0].set_ylabel('Precision@k')
axes[0].set_title('Precision@k — Structural vs Text vs Hybrid')
axes[0].legend()
axes[0].grid(True)
axes[1].set_xlabel('k')
axes[1].set_ylabel('Recall@k')
axes[1].set_title('Recall@k — Structural vs Text vs Hybrid')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_at_k.png'), dpi=150)
plt.close()
print("Saved precision_recall_at_k.png")


print("\nDone. Q3 output files in:", OUTPUT_DIR)
for f in ['user_vectors2026.data', 'precision_recall_at_k.png']:
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(fp):
        print("  %-40s %10d bytes" % (f, os.path.getsize(fp)))
