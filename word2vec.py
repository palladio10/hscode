import gensim
import re

TOKEN_RE = re.compile('[A-Za-z0-9]+')

sentences = []
with open("./sentences/per_receipt.txt", mode='r') as f:
    for line in f:
        sentences.append(TOKEN_RE.findall(line.lower()))
'''with open("./translation/fr_en.csv", mode='r') as f:
    for line in f:
        sentences.append(line.lower().split(","))
with open("./translation/rw_en.csv", mode='r') as f:
    for line in f:
        sentences.append(line.lower().replace('"', '').replace("\t", '').replace("\n", '').split(" "))'''

print(len(sentences))

num_features = 300    # Word vector dimensionality
min_word_count = 0   # Minimum word count
num_workers = 8       # Number of threads to run in parallel
context = 1           # Context window size
downsampling = 0.01    # Downsample setting for frequent words

# train word2vec on the two sentences
#model = gensim.models.Word2Vec(sentences, min_count=10, size=10, workers=4)
model = gensim.models.Word2Vec(sentences, workers=num_workers,
                               size=num_features, min_count = min_word_count,
                               window = context, sample = downsampling)
model.save("receipt_word2vec")
