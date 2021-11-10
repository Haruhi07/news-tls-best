import numpy as np
import datetime
import collections
import markov_clustering as mc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from scipy import sparse
from typing import List
from news_tls import utils, data, summarizers
from sentence_transformers import SentenceTransformer, util
import gensim
from gensim.models import HdpModel, LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer # to perform lemmatization or stemming in our pre-processing
from nltk.stem.porter import *


class TopicModeller():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, collection):
        articles = list(collection.articles())
        texts = ['{} {}'.format(a.title, a.text) for a in articles]
        result = []
        for text in texts:
            tmp = []
            for token in gensim.utils.simple_preprocess(text):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    tmp.append(self.lemmatizer.lemmatize(token, 'v'))
            result.append(tmp)
        return result

    def LDA(self, collection):
        texts = self.preprocess(collection)
        dictionary = gensim.corpora.Dictionary(texts)
        bow_corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = LdaModel(bow_corpus,
                             num_topics=20,
                             id2word=dictionary,
                             passes=4,
        )
        return lda_model

    def HDP(self, collection):
        texts = self.preprocess(collection)
        dictionary = gensim.corpora.Dictionary(texts)
        bow_corpus = [dictionary.doc2bow(text) for text in texts]
        hdp_model = HdpModel(bow_corpus,
                             id2word=dictionary
                    )
        return hdp_model

################################# Timeline Generator ###################################

class ClusteringTimelineGenerator():
    def __init__(self,
                 clusterer=None,
                 cluster_ranker=None,
                 summarizer=None,
                 clip_sents=5,
                 key_to_model=None,
                 unique_dates=True):

        self.clusterer = clusterer
        self.cluster_ranker = cluster_ranker
        self.summarizer = summarizer
        self.key_to_model = key_to_model
        self.unique_dates = unique_dates
        self.clip_sents = clip_sents

    def predict(self,
                collection,
                max_dates=10,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):
        #Topic Model here
        #print('lda topics...')
        lda_model = TopicModeller().LDA(collection)
        lda_topics = lda_model.print_topics()
        #for it in lda_topics:
        #    print(it)
        print('hdp topics...')
        hdp_model = TopicModeller().HDP(collection)
        hdp_topics = hdp_model.print_topics()
        #for it in hdp_topics:
        #    print(it)
        print('clustering articles...')

        # word embedding & cluster
        vectorizer = None
        embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')
        clusters = self.clusterer.cluster(collection, None, embedder)
        #doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        #clusters = self.clusterer.cluster(collection, doc_vectorizer, None)
        clusters_num = len(clusters)

        # calculate tfidf for all topics
        topic_model = lda_model
        def get_topic_words():
            topics = topic_model.print_topics()
            v_list = []
            w_list = []
            for i, topic in topics:
                terms = topic.split('+')
                tmp_v = []
                tmp_w = []
                for term in terms:
                    v, w = term.replace('\'', '').replace('\"', '').strip().split('*')
                    tmp_v.append(float(v))
                    tmp_w.append(w)
                v_list.append(tmp_v)
                w_list.append(tmp_w)
            return v_list, w_list

        def make_vocab(topic_words):
            ret = dict()
            for topic in topic_words:
                for word in topic:
                    if word in ret.keys():
                        continue
                    idx = len(ret)
                    ret[word] = idx
            return ret

        v, w = get_topic_words()
        n_topics = len(v)
        for topic_id in range(n_topics):
            row_sum = sum(v[topic_id])
            for word_id in range(len(v[topic_id])):
                v[topic_id][word_id] = v[topic_id][word_id] / row_sum
        vocab = make_vocab(w)
        n_vocab = len(vocab)

        def count_word():
            ret = np.zeros((n_topics, n_vocab))
            for topic_id in range(n_topics):
                for word_id, word in enumerate(w[topic_id]):
                    ret[topic_id][vocab[word]] = v[topic_id][word_id]
            return ret

        term_topic_matrix = count_word()

        def document_frequency(term_index):
            df = 0
            for d in range(n_topics):
                if term_topic_matrix[d, term_index] != 0:
                    df = df + 1
            return df

        def inverse_document_frequency(df):
            idf = np.log((n_topics + 1) / (df + 1)) + 1
            return idf

        tfidf_matrix = np.zeros((n_topics, n_vocab))

        for t in vocab:
            t_index = vocab[t]
            df = document_frequency(t_index)

            for d in range(n_topics):
                tf = term_topic_matrix[d, t_index]
                idf = inverse_document_frequency(df)
                tfidf_matrix[d, t_index] = tf * idf

        for d in range(n_topics):
            norm = np.sum(tfidf_matrix[d, :] ** 2)
            norm = np.sqrt(norm)
            tfidf_matrix[d, :] = tfidf_matrix[d, :] / norm

        vocab_list = list(vocab.keys())
        keywords = []
        for topic_id in range(n_topics):
            row = np.array(tfidf_matrix[topic_id, :])
            max_id = row.argsort()[-3:][::-1]
            tmp_word = [vocab_list[id] for id in max_id]
            keywords.append(tmp_word)
        print(keywords)

        topics = lda_topics
        centroid_list = [c.centroid for c in clusters]
        weighted_sim = np.zeros((len(topics), clusters_num))
        for j, topic in topics:
            tmp = topic.split('+')
            v_list = []
            w_list = []
            for t in tmp:
                v, w = t.replace('\'', '').replace('\"', '').strip().split('*')
                v_list.append(v)
                w_list.append(w)
            v_list = np.array(v_list, dtype=float)
            w_list = embedder.encode(w_list)
            unweighted_centroid_word_sim = cosine_similarity(centroid_list, w_list)
            weighted_sim[j] = np.matmul(unweighted_centroid_word_sim, v_list)

        weighted_sim = weighted_sim.transpose()
        #print('weighted similarity between topics and clusters...')
        #print(weighted_sim)

        # assign dates
        print('assigning cluster times...')
        for c in clusters:
            c.time = c.most_mentioned_time()
            if c.time is None:
                c.time = c.earliest_pub_time()

        print('ranking clusters...')
        ranked_clusters = self.cluster_ranker.rank(clusters, collection)

        print('vectorizing sentences...')

        def sent_filter(sent):
            return True

        print('summarization...')
        del lda_model, hdp_model, embedder
        sys_l = 0
        sys_m = 0
        ref_m = max_dates * max_summary_sents

        date_to_summary = collections.defaultdict(list)
        for c in ranked_clusters:

            date = c.time.date()
            c_sents = self._select_sents_from_cluster(c)
            #print("C", date, len(c_sents), "M", sys_m, "L", sys_l)
            summary = self.summarizer.summarize(c_sents)

            if summary:
                if self.unique_dates and date in date_to_summary:
                    continue
                date_to_summary[date] += summary
                sys_m += len(summary)
                if self.unique_dates:
                    sys_l += 1

            if sys_m >= ref_m or sys_l >= max_dates:
                break

        timeline = []
        for d, summary in date_to_summary.items():
            t = datetime.datetime(d.year, d.month, d.day)
            timeline.append((t, summary))
        timeline.sort(key=lambda x: x[0])

        return data.Timeline(timeline), clusters_num

    def _select_sents_from_cluster(self, cluster):
        sents = []
        for a in cluster.articles:
            for s in a.sentences[:self.clip_sents]:
            #for s in a.sentences:
                sents.append(s)
        return sents

    def load(self, ignored_topics):
        pass


################################# CLUSTERING ###################################


class Cluster:
    def __init__(self, articles, vectors, centroid, time=None, id=None):
        self.articles = sorted(articles, key=lambda x: x.time)
        self.centroid = centroid
        self.id = id
        self.vectors = vectors
        self.time = time

    def __len__(self):
        return len(self.articles)

    def pub_times(self):
        return [a.time for a in self.articles]

    def earliest_pub_time(self):
        return min(self.pub_times())

    def most_mentioned_time(self):
        mentioned_times = []
        for a in self.articles:
            for s in a.sentences:
                if s.time and s.time_level == 'd':
                    mentioned_times.append(s.time)
        if mentioned_times:
            return collections.Counter(mentioned_times).most_common()[0][0]
        else:
            return None

    def update_centroid(self):
        #X = sparse.vstack(self.vectors)
        #self.centroid = sparse.csr_matrix.mean(X, axis=0)
        X = np.vstack(self.vectors)
        self.centroid = np.mean(X, axis=0)


class Clusterer():
    def cluster(self, collection, vectorizer, embedder) -> List[Cluster]:
        raise NotImplementedError


class OnlineClusterer(Clusterer):
    def __init__(self, max_days=1, min_sim=0.5):
        self.max_days = max_days
        self.min_sim = min_sim

    def cluster(self, collection, vectorizer, embedder) -> List[Cluster]:
        # build article vectors
        texts = ['{} {}'.format(a.title, a.text) for a in collection.articles]
        try:
            X =  vectorizer.transform(texts)
        except:
            X = vectorizer.fit_transform(texts)

        id_to_vector = {}
        for a, x in zip(collection.articles(), X):
            id_to_vector[a.id] = x

        online_clusters = []

        for t, articles in collection.time_batches():
            for a in articles:

                # calculate similarity between article and all clusters
                x = id_to_vector[a.id]
                cluster_sims = []
                for c in online_clusters:
                    if utils.days_between(c.time, t) <= self.max_days:
                        centroid = c.centroid
                        sim = cosine_similarity(centroid, x)[0, 0]
                        cluster_sims.append(sim)
                    else:
                        cluster_sims.append(0)

                # assign article to most similar cluster (if over threshold)
                cluster_found = False
                if len(online_clusters) > 0:
                    i = np.argmax(cluster_sims)
                    if cluster_sims[i] >= self.min_sim:
                        c = online_clusters[i]
                        c.vectors.append(x)
                        c.articles.append(a)
                        c.update_centroid()
                        c.time = t
                        online_clusters[i] = c
                        cluster_found = True

                # initialize new cluster if no cluster was similar enough
                if not cluster_found:
                    new_cluster = Cluster([a], [x], x, t)
                    online_clusters.append(new_cluster)

        clusters = []
        for c in online_clusters:
            cluster = Cluster(c.articles, c.vectors)
            clusters.append(cluster)

        return clusters


class TemporalMarkovClusterer(Clusterer):
    def __init__(self, max_days=1):
        self.max_days = max_days

    def cluster(self, collection, vectorizer, embedder) -> List[Cluster]:
        articles = list(collection.articles())
        # word embedding
        if vectorizer != None:
            texts = ['{} {}'.format(a.title, a.text) for a in articles]
            try:
                X = vectorizer.transform(texts)
            except:
                X = vectorizer.fit_transform(texts)
        else:
            texts = list()
            for a in articles:
                tmp_text = list()
                if a.title:
                    tmp_text.append(a.title)
                sents = a.text.split('\n')
                for sent in sents:
                    if sent != b'':
                        tmp_text.append(sent)
                sent_embed = embedder.encode(tmp_text)
                texts.append(np.mean(sent_embed, axis=0)) #use average sentence embeddings as document embedding

            X = np.vstack(texts)


        f = open("clust_result.txt", "w")
        print("--------------", file=f)

        times = [a.time for a in articles]
        print(times, file=f)

        print('temporal graph...')
        S = self.temporal_graph(X, times)

        print('run markov clustering...')
        result = mc.run_mcl(S)
        print('done')

        idx_clusters = mc.get_clusters(result)
        idx_clusters.sort(key=lambda c: len(c), reverse=True)

        print(f'times: {len(set(times))} articles: {len(articles)} '
              f'clusters: {len(idx_clusters)}')


        clusters = []
        for c in idx_clusters:
            print(c, file=f)
            c_vectors = [X[i] for i in c]
            c_articles = [articles[i] for i in c]
            for a in c_articles:
                print(a.time, file=f)
            #Xc = sparse.vstack(c_vectors)
            #centroid = sparse.csr_matrix(Xc.mean(axis=0))
            Xc = np.vstack(c_vectors)
            centroid = np.mean(Xc, axis=0)
            cluster = Cluster(c_articles, c_vectors, centroid=centroid)
            #print(c_articles, centroid, file=f)
            clusters.append(cluster)

        f.close()

        return clusters

    def temporal_graph(self, X, times):
        times = [utils.strip_to_date(t) for t in times]
        time_to_ixs = collections.defaultdict(list)
        for i in range(len(times)):
            time_to_ixs[times[i]].append(i)


        n_items = X.shape[0]

        #S = sparse.lil_matrix((n_items, n_items))
        S = np.zeros((n_items, n_items))
        start, end = min(times), max(times)
        total_days = (end - start).days + 1

        for n in range(total_days + 1):
            t = start + datetime.timedelta(days=n)
            window_size = min(self.max_days + 1, total_days + 1 - n)
            window = [t + datetime.timedelta(days=k) for k in range(window_size)]


            if n == 0 or len(window) == 1:
                indices = [i for t in window for i in time_to_ixs[t]]
                if len(indices) == 0:
                    continue

                #X_n = sparse.vstack([X[i] for i in indices])
                X_n = np.vstack([X[i] for i in indices])
                S_n = cosine_similarity(X_n)
                n_items = len(indices)
                for i_x, i_n in zip(indices, range(n_items)):
                    for j_x, j_n in zip(indices, range(i_n + 1, n_items)):
                        S[i_x, j_x] = S_n[i_n, j_n]
            else:
                # prev is actually prev + new
                prev_indices = [i for t in window for i in time_to_ixs[t]]
                new_indices = time_to_ixs[window[-1]]

                if len(new_indices) == 0:
                    continue

                #X_prev = sparse.vstack([X[i] for i in prev_indices])
                #X_new = sparse.vstack([X[i] for i in new_indices])
                X_prev = np.vstack([X[i] for i in prev_indices])
                X_new = np.vstack([X[i] for i in new_indices])
                S_n = cosine_similarity(X_prev, X_new)
                n_prev, n_new = len(prev_indices), len(new_indices)
                for i_x, i_n in zip(prev_indices, range(n_prev)):
                    for j_x, j_n in zip(new_indices, range(n_new)):
                        S[i_x, j_x] = S_n[i_n, j_n]

        #return sparse.csr_matrix(S)
        return S

class AffinityPropagationClusterer(Clusterer):
    def __init__(self, max_days=1):
        self.max_days = max_days

    def cluster(self, collection, vectorizer, embedder) -> List[Cluster]:
        articles = list(collection.articles())

        if vectorizer != None:
            texts = ['{} {}'.format(a.title, a.text) for a in articles]
            X = vectorizer.fit_transform(texts)
        else:
            texts = list()
            for a in articles:
                tmp_text = list()
                if a.title:
                    tmp_text.append(a.title)
                sents = a.text.split('\n')
                for sent in sents:
                    if sent != b'':
                        tmp_text.append(sent)
                sent_embed = embedder.encode(tmp_text)
                texts.append(np.mean(sent_embed, axis=0)) #use average sentence embeddings as document embedding

            X = np.vstack(texts)

        times = [a.time for a in articles]

        def calculate_similarity(method = 'euclid'):
            if method == 'euclid':
                S = np.zeros((len(X), len(X)))
                for i in range(len(X)):
                    for j in range(i, len(X)):
                        S[i][j] = - sum((X[i] - X[j]) ** 2)
                        S[j][i] = S[i][j]
            elif method == 'cosine':
                S = cosine_similarity(X) - 1

            for i, time_i in enumerate(times):
                for j, time_j in enumerate(times):
                    time_gap = max(time_i, time_j) - min(time_i, time_j)
                    if time_gap > datetime.timedelta(days=1):
                        S[i][j] = -100
            return S

        S = calculate_similarity('euclid')
        af = AffinityPropagation(preference=-50, affinity='precomputed', random_state=None).fit(S)
        cluster_centers = af.cluster_centers_indices_
        labels = af.labels_

        if labels[0] == -1:
            print('cosine')
            S = calculate_similarity('cosine')
            af = AffinityPropagation(preference=-50, affinity='precomputed', random_state=None).fit(S)
            cluster_centers = af.cluster_centers_indices_
            labels = af.labels_

        print(f'times: {len(set(times))} articles: {len(articles)} '
              f'clusters: {len(set(labels))}')

        print(labels)
        print(cluster_centers)

        idx_clusters = collections.defaultdict(list)
        for i in range(len(X)):
            idx_clusters[cluster_centers[labels[i]]].append(i)
        for c in idx_clusters:
            print('{} {}'.format(c, idx_clusters[c]))

        clusters = []
        for c in idx_clusters:
            c_vectors = [X[i] for i in idx_clusters[c]]
            c_articles = [articles[i] for i in idx_clusters[c]]
            Xc = np.vstack(c_vectors)
            #centroid = np.mean(Xc, axis=0)
            centroid = X[c]
            cluster = Cluster(c_articles, c_vectors, centroid=centroid)
            clusters.append(cluster)

        return clusters



############################### CLUSTER RANKING ################################


class ClusterRanker:
    def rank(self, clusters, collection, vectorizer):
        raise NotImplementedError


class ClusterSizeRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None):
        return sorted(clusters, key=len, reverse=True)


class ClusterDateMentionCountRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None):
        date_to_count = collections.defaultdict(int)
        for a in collection.articles():
            for s in a.sentences:
                d = s.get_date()
                if d:
                    date_to_count[d] += 1

        clusters = sorted(clusters, reverse=True, key=len)

        def get_count(c):
            t = c.most_mentioned_time()
            if t:
                return date_to_count[t.date()]
            else:
                return 0

        clusters = sorted(clusters, reverse=True, key=get_count)
        return sorted(clusters, key=len, reverse=True)














#
