

sentences = list(df[TEXT_COLUMN].fillna(''))
sentences = [s.split(' ') for s in sentences]
#sentences = [['first', 'sentence'], ['second', 'sentence']]




num_features = 300    # Word vector dimensionality
min_word_count = 10   # Minimum word count
num_workers = 8       # Number of threads to run in parallel
context = 1           # Context window size
downsampling = 0.01    # Downsample setting for frequent words

# train word2vec on the two sentences
#model = gensim.models.Word2Vec(sentences, min_count=10, size=10, workers=4)
model = gensim.models.Word2Vec(sentences, workers=num_workers,
                               size=num_features, min_count = min_word_count,
                               window = context, sample = downsampling)