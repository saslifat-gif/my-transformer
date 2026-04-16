import json

with open('/Users/lifat/ml/my_trans/zh_wiki_clean.txt', "r", encoding = 'utf-8') as f:
    text = f.read()
tokens = list(text.encode("utf-8"))

vocab_size = 5000

#----------BPE tokenizer----------

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] ==ids[i + 1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
    
def bpe_train(ids, vocab_size):
    merges = {}
    merge_times = vocab_size - 256
    for i in range(merge_times):
        stats = get_stats(ids)
        pair = max(stats, key = stats.get)
        idx = i + 256
        print(f"{i} merging {pair} in to {idx}")
        ids = merge(ids, pair, idx)
        merges[idx] = pair
    return merges, ids

merges, ids = bpe_train(tokens, vocab_size)
merges_serialzable = {str()}
vocab = {idx: bytes([idx]) for idx in range(256)}
merges_serializable = {str(k): list(v) for k, v in merges.items()}
with open(f'zh_wiki_merges{vocab_size}.json', 'w') as f:
    json.dump(merges_serializable, f)

# save encoded ids
with open(f'zh_wiki_ids{vocab_size}.json', 'w') as f:
    json.dump(ids, f)

print("saved!")

print(f"raw data length {len(tokens)}")
print(f"new data length {len(ids)}")
print(f"ration = {len(tokens)/len(ids):.2f}x")