from flask import Flask, request, jsonify, render_template, session
from bs4 import BeautifulSoup
from requests import get
import requests
from newspaper import Article
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import pickle
import json
import sys
import os

app = Flask(__name__)
app.secret_key = 'any random string'

def scrape_news(keywords):

    keywords_list = keywords.split()
    url = 'https://search.naver.com/search.naver?where=news&query='
    for word in keywords_list:
        if keywords_list.index(word)==0:
            url = url + word
        else:
            url = url + '+' + word

    response = get(url)

    url_list = []

    html_soup = BeautifulSoup(response.text, 'html.parser')
    for a in html_soup.find_all('a', class_ = ' _sp_each_title'):
        url_list.append(a['href'])

    print(url_list)
    sys.stdout.flush()

    news = []
    titles = []
    urls = []
    for url in url_list:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            print(text)
            sys.stdout.flush()
            if len(text.replace('\n', ' ').replace('?','.').replace('!','.').split('.'))>10:
                urls.append(url)
                titles.append(article.title)
                news.append(text.replace('\n', ' ').replace('?','.').replace('!','.').split('.'))
        except:
            pass

    return titles, news, urls

def news_vectorization(input_text):

    sent_vectors = []
    article_vectors = []

    sent_model = Doc2Vec.load("./model/d2v_politics_sentence.model")

    for article in input_text:
        article_text = article
        vec_list = []
        for sent in article_text:
            vec = sent_model.infer_vector(sent.split(' '), epochs=5000)
            vec_list.append(vec)
            sent_vectors.append(vec)
        article_vectors.append(vec_list)

    return sent_vectors, article_vectors


def news_summarization(input_text ,article_vectors):

    summarize_news = []
    for n, article_vec in enumerate(article_vectors):
        summarize_kmeans = KMeans(n_clusters=5, random_state=0).fit(article_vec)
        summarize_vecs = []
        for j, sent_vec in enumerate(article_vec):
            summarize_vecs.append(sent_vec)
        summarize_article=[]
        for i in range(5):
            cosine_score = [c[0] for c in cosine_similarity(summarize_vecs, [summarize_kmeans.cluster_centers_[i]])]
            summarize_sents = input_text[n][cosine_score.index(max(cosine_score))]
            summarize_article.append(summarize_sents)
        summarize_article = [x for x in list(set(summarize_article))]
        summarize_indices = []
        for sent in summarize_article:
            summarize_indices.append(input_text[n].index(sent))
        summarize_article = '. '.join([input_text[n][i] for i in sorted(summarize_indices)])
        summarize_news.append(summarize_article)

    return summarize_news


def sentiment_classification(summarize_news, sent_vectors):

    input_sents = []

    for s in summarize_news:
        input_sents = input_sents + s.split('.')
    print(input_sents)
    sys.stdout.flush()

    with open('./model/sentiment_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    sentiment_model = load_model('./model/sentiment_model.h5')
    sentiment_sents = []
    sentiment_vectors = []

    predict = sentiment_model.predict(sequence.pad_sequences(tokenizer.texts_to_sequences(input_sents), maxlen=128))

    for n, pred in enumerate(predict):
        print(n, input_sents[n][:30], pred.argmax(), pred)
        if pred[1] > 0.005:
            sentiment_sents.append(input_sents[n])
            sentiment_vectors.append(sent_vectors[n])
        else:
            pass

    print(sentiment_sents)
    sys.stdout.flush()
    del sentiment_model
    K.clear_session()

    return sentiment_sents, sentiment_vectors


def named_entity_recognition(input_text):

    model = load_model('./model/ner_model.h5', custom_objects={'CRF': CRF, 'crf_loss':crf_loss, 'crf_viterbi_accuracy':crf_viterbi_accuracy})

    with open('./model/ner_src_tokenizer.pickle', 'rb') as handle:
        src_tokenizer = pickle.load(handle)

    with open('./model/ner_tar_tokenizer.pickle', 'rb') as handle:
        tar_tokenizer = pickle.load(handle)

    entity_data = []

    for text in input_text:
        sentences= text
        entity = []

        for sent in sentences:
            new_sentence = sent.split()

            word_to_index = src_tokenizer.word_index
            new_X=[]
            for w in new_sentence:
                try:
                    new_X.append(word_to_index.get(w,1))
                except KeyError:
                    new_X.append(word_to_index['OOV'])

            pad_new = pad_sequences([new_X], padding="post", value=0, maxlen=128)
            p = model.predict(np.array([pad_new[0]]))
            p = np.argmax(p, axis=-1)

            for w, pred in zip(new_sentence, p[0]):
                label = tar_tokenizer.index_word[pred]
                if label != 'O':
                    entity.append((w, label))

        entity_dict = {'인물': '인물: {}'.format(', '.join(list(set([x[0] for x in set(entity) if len(x[0])>1 and len(x[0])<=3 and x[1]=='인물'])))),
        '정당' : '정당: {}'.format(', '.join(list(set([x[0] for x in set(entity) if len(x[0])>1 and x[1]=='정당'])))),
        '기관/집단' : '기관/집단: {}'.format(', '.join(list(set([x[0] for x in set(entity) if len(x[0])>1 and x[1]=='기관/집단'])))),
        '장소' : '장소: {}'.format(', '.join(list(set([x[0] for x in set(entity) if len(x[0])>1 and x[1]=='장소']))))}

        entity_data.append(entity_dict)

    del model
    K.clear_session()

    return entity_data

def topic_vectorization(politics_db, news_topic, sentiment_vectors):

    kmeans = pickle.load(open("./model/kmeans_d2v.pkl", "rb"))
    sent_model = Doc2Vec.load("./model/d2v_politics_sentence.model")
    news_model =  Doc2Vec.load("./model/d2v_politics_news.model")

    topic_vectors = []
    for i in politics_db.execute("SELECT doc2vec_index  FROM politicsDB WHERE topic = {};".format(news_topic)):
        topic_vectors.append((news_model.docvecs[i[0]],i[0]))
    topic_scores = [np.sum(j) for j in cosine_similarity([x[0] for x in topic_vectors], [kmeans.cluster_centers_[news_topic]])]
    topic_indices = [topic_vectors[topic_scores.index(x)][1] for x in sorted(topic_scores,reverse=True)[:50]]

    fin_scores = [np.sum(j) for j in cosine_similarity([sent_model.docvecs[str(i)] for i in topic_indices], sentiment_vectors)]
    fin_indices = [topic_indices[t] for t in [fin_scores.index(n) for n in sorted(fin_scores, reverse=True)[:5]]]

    return fin_indices


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/input_loading", methods=["POST"])
def input_loading():

    keywords = request.form.get("keywords")
    session['keywords'] = keywords
    print(keywords)
    sys.stdout.flush()

    return render_template("input_loading.html", keywords=keywords)

@app.route("/analysis_page")
def analysis_page():

    keywords = session['keywords']
    titles, news, urls = scrape_news(keywords)

    print(news)
    sys.stdout.flush()

    ner_data = named_entity_recognition(news)
    sent_vectors, article_vectors = news_vectorization(news)
    summarize_news = news_summarization(news, article_vectors)

    sentiment_sents, sentiment_vectors = sentiment_classification(summarize_news, sent_vectors)
    print(sentiment_sents)
    sys.stdout.flush()

    kmeans = pickle.load(open("./model/kmeans_d2v.pkl", "rb"))

    news_vectors = []
    news_model = Doc2Vec.load("./model/d2v_politics_news.model")
    for article in news:
        news_vectors.append(news_model.infer_vector('.'.join(article).split(), epochs=500))

    scores = [np.sum(x) for x in cosine_similarity(kmeans.cluster_centers_, news_vectors)]
    k = scores.index(max(scores))
    session['topic'] = k

    K.clear_session()

    return render_template("analysis_page.html",  enumerate=enumerate, titles=titles, urls=urls, ner_data=ner_data, summarize_news=summarize_news, sentiment_sents=sentiment_sents)

@app.route("/recommendations_loading", methods=["POST"])
def recommendations_loading():

    print('recommendations_loading')
    sys.stdout.flush()
    political_preference = request.form.getlist('pr')
    print(political_preference)
    sys.stdout.flush()
    session['political_preference'] = political_preference
    print(session['political_preference'])
    sys.stdout.flush()

    return render_template("recommendations_loading.html")

@app.route("/recommendations_page")
def recommendations_page():

    political_preference = session['political_preference']
    print(political_preference)
    sys.stdout.flush()

    sentiment_vectors = []
    sent_model = Doc2Vec.load("./model/d2v_politics_sentence.model")
    for sent in political_preference:
        vec = sent_model.infer_vector(sent.split(' '), epochs=5000)
        sentiment_vectors.append(vec)

    engine = create_engine('mysql+pymysql://root:wjdqkf20322@34.64.92.31/politics_database')
    engine.connect()
    politics_db = scoped_session(sessionmaker(bind=engine))

    print('connected to db')
    sys.stdout.flush()

    news_topic = int(session['topic'])

    fin_indices = topic_vectorization(politics_db, news_topic, sentiment_vectors)

    sentiment_sentences = []
    date = []
    titles = []
    articles = []
    urls=[]
    for i in fin_indices:
        for p in politics_db.execute("SELECT * FROM politicsDB WHERE doc2vec_index = {};".format(int(i))):
            titles.append(p[1])
            urls.append(p[4])
            date.append(p[3])
            articles.append(p[2])
            sentiment_sentences.append(p[5])

    news = []
    for a in articles:
        if len(a.replace('?','.').replace('!','.').split('.'))>10:
            news.append(a.replace('?','.').replace('!','.').split('.'))
    print(news)
    sys.stdout.flush()
    ner_data = named_entity_recognition(news)
    sent_vectors, article_vectors = news_vectorization(news)
    summarize_news = news_summarization(news, article_vectors)
    K.clear_session()

    return render_template("recommendations_page.html", zip=zip, sentiment_sentences=sentiment_sentences, summarize_news=summarize_news, date=date, urls=urls, ner_data=ner_data, titles=titles)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
