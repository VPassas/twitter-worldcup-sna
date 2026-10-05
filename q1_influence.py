import json
import os
import csv
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'twitter_world_cup_part1')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'project_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#read tweets
print("Loading tweets...")

files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('tweets.json.')],
               key=lambda x: int(x.split('.')[-1]))

corpus        = []
user_tweets   = defaultdict(list)
user_hashtags = defaultdict(list)
user_names    = {}
edges         = []
nTweets       = 0

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
                corpus.append(tw.get('text', ''))
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
                    edges.append((uid, tid))
                reply_id = tw.get('in_reply_to_user_id')
                if reply_id is not None:
                    reply_id = str(reply_id)
                    user_names[reply_id] = tw.get('in_reply_to_screen_name', '')
                    edges.append((uid, reply_id))
                if 'retweeted_status' in tw:
                    rt_uid = str(tw['retweeted_status']['user']['id'])
                    user_names[rt_uid] = tw['retweeted_status']['user']['screen_name']
                    edges.append((uid, rt_uid))
                nTweets += 1
            except:
                pass
        c = infile.readline()
    infile.close()

print("Total tweets loaded:", nTweets)
print('Number of Tweets=', len(corpus))
print('Unique users:', len(user_names))
print('Directed edges (raw):', len(edges))



print("\nQ1 - Influence")

# build directed user to user network
print("Building directed network...")
G = nx.DiGraph()
for src, tgt in edges:
    if src == tgt:
        continue
    if G.has_edge(src, tgt):
        G[src][tgt]['weight'] += 1
    else:
        G.add_edge(src, tgt, weight=1)

print('#nodes=', G.number_of_nodes())
print('#edges=', G.number_of_edges())

netfile = os.path.join(OUTPUT_DIR, 'network-Passas2026.data')
nx.write_edgelist(G, netfile, data=['weight'])
print("Saved", netfile)

G_und = G.to_undirected()

print('Size of connected components')
comp = list(nx.connected_components(G_und))
print('#Connected components', len(comp))
for compLen in comp[:5]:
    print('size=', len(compLen))

# degree distribution log-log with OLS slope
in_degs  = [d for _, d in G.in_degree()]
out_degs = [d for _, d in G.out_degree()]

max_in  = max(in_degs)  if in_degs  else 0
max_out = max(out_degs) if out_degs else 0

hg_in  = [in_degs.count(i)  for i in range(max_in  + 1)]
hg_out = [out_degs.count(i) for i in range(max_out + 1)]

hg_in_nz  = [x if x > 0 else 1 for x in hg_in]
hg_out_nz = [x if x > 0 else 1 for x in hg_out]

in_deg_pos  = [(d, c) for d, c in enumerate(hg_in)  if d > 0 and c > 0]
out_deg_pos = [(d, c) for d, c in enumerate(hg_out) if d > 0 and c > 0]

x_in,  y_in  = np.log([d for d, _ in in_deg_pos]),  np.log([c for _, c in in_deg_pos])
x_out, y_out = np.log([d for d, _ in out_deg_pos]), np.log([c for _, c in out_deg_pos])

slope_in,  intercept_in  = np.polyfit(x_in,  y_in,  1)
slope_out, intercept_out = np.polyfit(x_out, y_out, 1)

print('power-law investigation (log-log linear fit):')
print('  in-degree:  slope=%.3f  (alpha~%.3f)' % (slope_in,  -slope_in))
print('  out-degree: slope=%.3f  (alpha~%.3f)' % (slope_out, -slope_out))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_in_plot  = np.log(np.arange(1, len(hg_in_nz)  + 1))
y_in_sorted  = sorted(np.log(hg_in_nz),  reverse=True)
axes[0].plot(x_in_plot, y_in_sorted, '.')
axes[0].plot(x_in, slope_in * x_in + intercept_in, '--', color='gray', lw=1,
             label='OLS slope=%.2f' % slope_in)
axes[0].set_xlabel('In-Degree (log scale)')
axes[0].set_ylabel('Number of nodes (log scale)')
axes[0].set_title('In-Degree Distribution (log-log)')
axes[0].legend(fontsize=8)

x_out_plot = np.log(np.arange(1, len(hg_out_nz) + 1))
y_out_sorted = sorted(np.log(hg_out_nz), reverse=True)
axes[1].plot(x_out_plot, y_out_sorted, '.')
axes[1].plot(x_out, slope_out * x_out + intercept_out, '--', color='gray', lw=1,
             label='OLS slope=%.2f' % slope_out)
axes[1].set_xlabel('Out-Degree (log scale)')
axes[1].set_ylabel('Number of nodes (log scale)')
axes[1].set_title('Out-Degree Distribution (log-log)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'degree_distributions.png'), dpi=150)
plt.close()
print("Saved degree_distributions.png")


# centrality measures
print("Computing centrality measures...")

in_cen  = nx.in_degree_centrality(G)
out_cen = nx.out_degree_centrality(G)
pgNodes = nx.pagerank(G)

# Exact betweenness on top-10 in-degree subgraph
print("Computing exact betweenness on top-10 in-degree subgraph...")
top10_ids = [uid for uid, _ in sorted(in_cen.items(), key=lambda x: x[1], reverse=True)[:10]]
G_sub     = G.subgraph(top10_ids)
print('Subgraph: %d nodes, %d edges' % (G_sub.number_of_nodes(), G_sub.number_of_edges()))
btNodes = nx.betweenness_centrality(G_sub, normalized=True)

with open(os.path.join(OUTPUT_DIR, 'degree.data'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'screen_name', 'in_degree_centrality', 'out_degree_centrality'])
    sorted_deg = sorted(in_cen.items(), key=lambda x: x[1] + out_cen[x[0]], reverse=True)
    for uid, val in sorted_deg:
        w.writerow([uid, user_names.get(uid, ''), round(in_cen[uid], 8), round(out_cen[uid], 8)])
print("Saved degree.data")

with open(os.path.join(OUTPUT_DIR, 'pageRank.data'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'screen_name', 'pagerank'])
    for uid, val in sorted(pgNodes.items(), key=lambda x: x[1], reverse=True):
        w.writerow([uid, user_names.get(uid, ''), val])
print("Saved pageRank.data")

with open(os.path.join(OUTPUT_DIR, 'betweeness.data'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'screen_name', 'betweenness'])
    for uid, val in sorted(btNodes.items(), key=lambda x: x[1], reverse=True):
        w.writerow([uid, user_names.get(uid, ''), val])
print("Saved betweeness.data")


# top 10 influential users ranked by in-degree, compared across all metrics
print("\nTop 10 influential users (ranked by in-degree centrality):")

top10_in = sorted(in_cen.items(), key=lambda x: x[1], reverse=True)[:10]
top10_pr = sorted(pgNodes.items(), key=lambda x: x[1], reverse=True)[:10]
top10_bt = sorted(btNodes.items(), key=lambda x: x[1], reverse=True)[:10]

wantIds = set(uid for uid, _ in top10_in)

topUserDocs   = {uid: ' '.join(user_tweets.get(uid, [])) for uid in wantIds}
topUserList   = list(topUserDocs.keys())
topUserCorpus = [topUserDocs[uid] for uid in topUserList]

my_stop_words_top = text.ENGLISH_STOP_WORDS.union([
    'http', 'https', 'amp', 'rt', 'co', 'via', 'www',
    'worldcup2014', 'worldcup', 'world', 'cup', 'brazil', '2014'])

try:
    topVec    = TfidfVectorizer(stop_words=list(my_stop_words_top), max_features=5000, min_df=1)
    topMatrix = topVec.fit_transform(topUserCorpus).toarray()
    topFeatures = np.array(topVec.get_feature_names_out())
    USE_TFIDF_TOP = True
except ValueError:
    USE_TFIDF_TOP = False

def get_tags(uid):
    if not USE_TFIDF_TOP or uid not in topUserDocs:
        return ['#' + t for t, _ in Counter(user_hashtags.get(uid, [])).most_common(5)]
    doc = topUserDocs[uid]
    if not doc.strip():
        return ['#' + t for t, _ in Counter(user_hashtags.get(uid, [])).most_common(5)]
    i = topUserList.index(uid)
    indexes = topMatrix[i, :].argsort()[-5:][::-1]
    return list(topFeatures[indexes])

# Centrality comparison table for top-10 in-degree users
print('%-5s %-20s %10s %10s %10s %10s' % ('rank', 'user', 'in_cen', 'out_cen', 'pagerank', 'between'))
for rank, (uid, ic) in enumerate(top10_in, 1):
    oc = out_cen.get(uid, 0)
    pr = pgNodes.get(uid, 0)
    bt = btNodes.get(uid, 0)
    tags = get_tags(uid)
    print('  %2d. %-20s %10.6f %10.6f %10.6f %10.6f  tags: %s' % (
        rank, user_names.get(uid, uid), ic, oc, pr, bt, tags))

set_in = set(uid for uid, _ in top10_in)
set_pr = set(uid for uid, _ in top10_pr)
set_bt = set(uid for uid, _ in top10_bt)

print('overlap in-degree & pagerank:',    len(set_in & set_pr))
print('overlap in-degree & betweenness:', len(set_in & set_bt))
print('overlap pagerank & betweenness:',  len(set_pr & set_bt))
print('in all three:', [user_names.get(u, '') for u in set_in & set_pr & set_bt])

print('top-10 in-degree users - neighbor topics:')
for uid, _ in top10_in:
    neighbors = set(G.successors(uid)) | set(G.predecessors(uid))
    nb_ht = []
    for nb in neighbors:
        nb_ht.extend(user_hashtags.get(nb, []))
    own = [t for t, _ in Counter(user_hashtags.get(uid, [])).most_common(3)]
    nbr = [t for t, _ in Counter(nb_ht).most_common(3)]
    print("  %s: own=%s, neighbors=%s" % (user_names.get(uid, uid), own, nbr))


print("\nDone. Q1 output files in:", OUTPUT_DIR)
for f in ['network-Passas2026.data', 'degree.data', 'pageRank.data',
          'betweeness.data', 'degree_distributions.png']:
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(fp):
        print("  %-40s %10d bytes" % (f, os.path.getsize(fp)))
