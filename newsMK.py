import requests as rq
from bs4 import BeautifulSoup
import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#spacy.cli.download('ru_core_news_sm')

stop_words = stopwords.words('russian')


#парсинг новостей общей тематики
url = "https://www.mk.ru/news/2024/6/9/"
page = rq.get(url)
soup = BeautifulSoup(page.text, features="html.parser")

news_title = soup.find_all('h3', {'class' : 'news-listing__item-title'})
text_title = [i.text for i in news_title]


links = []
for i in soup.find_all('a', {'class' : 'news-listing__item-link'}):
      links.append(i.get('href'))


def GetNewsBody(url):
  page = rq.get(url)
  soup = BeautifulSoup(page.text, features="html.parser")
  body_text = []
  for item in soup.find_all('div', {'class': 'article__body'}):
        for i in item.find_all('p'):
            body_text.append(i.text.strip())

  return ' '.join(body_text)

news_body = []
for link in links:
    new = GetNewsBody(link)
    news_body.append(new)


for i in news_body:
    i = i.replace('\xa0', ' ')


# print(news_body)
#
news_df = pd.DataFrame({'link': links, 'title': text_title, 'text': news_body})
# news_df.to_csv('newsMK_df.csv')
# print(news_df)

nlp_rus = spacy.load("ru_core_news_sm")
#сделаем отдельный столбец с очищенным и лемматизированным текстом
def preprocess(text):
    tokenized = word_tokenize(text)
    text_clean = []
    for word in tokenized:
        if word[0].isalnum() and word not in stop_words:
            text_clean.append(word)
    doc = nlp_rus(' '.join(text_clean)) #передаем в spacy и лемматизируем
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    return ' '.join(lemmas)

news_df['text_lemmas'] = news_df['text'].apply(preprocess)

#выделение именнованных сущностей для каждой статьи

ents = [] #список списков для всех NER по статьям
for i in news_df['text_lemmas']:
    doc = nlp_rus(i)
    article_ents = [] #список для NER одной статьи
    for ent in doc.ents:
        article_ents.append((ent.text, ent.label_))
    ents.append(article_ents)

#можно позже слепить именованную сущность и лейбл

news_df['NER'] = ents #добавляем колонку с NER в датафрейм
news_df.to_csv('newsMK_df.csv')

#предобработанный текст на тренировочную и тестовую выборку, чтобы иметь возможность оценить качество работы модели
X = news_df['text_lemmas']
y = news_df['title'].tolist() #y мы не будем использовать, нужен только чтобы поделить на тренировочное и тестовое множество


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=41
                                                    )

#векторизируем тексты
Tfidf = TfidfVectorizer()
Tfidf.fit(X_train)
X_train_vec = Tfidf.transform(X_train)
X_test_vec = Tfidf.transform(X_test)

#применем lda c разным количеством n_components и оценим модель, чтобы выбрать оптимальное количество групп

lda2 = LatentDirichletAllocation(n_components=2, random_state=0)
topics2 = lda2.fit_transform(X_train_vec)
print(lda2.perplexity(X_test_vec))
print(lda2.score(X_test_vec))

lda3 = LatentDirichletAllocation(n_components=3, random_state=0)
topics3 = lda3.fit_transform(X_train_vec)
print(lda3.perplexity(X_test_vec))
print(lda3.score(X_test_vec))

lda5 = LatentDirichletAllocation(n_components=5, random_state=0)
topics5 = lda5.fit_transform(X_train_vec)
print(lda5.perplexity(X_test_vec))
print(lda5.score(X_test_vec))

lda7 = LatentDirichletAllocation(n_components=7, random_state=0)
topics7 = lda7.fit_transform(X_train_vec)
print(lda7.perplexity(X_test_vec))
print(lda7.score(X_test_vec))

lda10 = LatentDirichletAllocation(n_components=10, random_state=0)
topics10 = lda10.fit_transform(X_train_vec)
print(lda10.perplexity(X_test_vec))
print(lda10.score(X_test_vec))


lda12 = LatentDirichletAllocation(n_components=12, random_state=0)
topics12 = lda12.fit_transform(X_train_vec)
print(lda12.perplexity(X_test_vec))
print(lda12.score(X_test_vec))


lda15 = LatentDirichletAllocation(n_components=15, random_state=0)
topics15 = lda15.fit_transform(X_train_vec)
print(lda15.perplexity(X_test_vec))
print(lda15.score(X_test_vec))

lda17 = LatentDirichletAllocation(n_components=17, random_state=0)
topics17 = lda17.fit_transform(X_train_vec)
print(lda17.perplexity(X_test_vec))
print(lda17.score(X_test_vec))


lda20 = LatentDirichletAllocation(n_components=20, random_state=0)
topics20 = lda20.fit_transform(X_train_vec)
print(lda20.perplexity(X_test_vec))
print(lda20.score(X_test_vec))


lda25 = LatentDirichletAllocation(n_components=15, random_state=0)
topics25 = lda25.fit_transform(X_train_vec)
print(lda25.perplexity(X_test_vec))
print(lda25.score(X_test_vec))

#чем ниже perplexity и чем ближе score к нулю, тем лучше. в нашем случае получилось, что чем меньше количество топиков тем лучше. Отрисуем самые значимые слова для моделей, где 2, 3 и 5  топиков

tf_feature_names = Tfidf.get_feature_names_out()
n_components2 = 2
n_components3 = 3
n_components5 = 5


def plot_top_words(model, feature_names, n_top_words, title, n_components, max_plots=5):
    fig, axes = plt.subplots(1, max_plots, figsize=(25, 10))  # параметры отображения
    axes = axes.flatten()
    all_features = {}  # словарь для сохранения ключевых слов для тем

    for topic_idx, topic in enumerate(model.components_):
        if topic_idx < max_plots:
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]

            # строка для сохранения темы и слов в словарь
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx + 1}',
                         fontdict={'fontsize': 13})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=10)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=14)

    plt.show()

# plot_top_words(lda2, tf_feature_names, 10, 'Распределение слов по темам, LDA-модель', n_components2)
# plot_top_words(lda3, tf_feature_names, 10, 'Распределение слов по темам, LDA-модель', n_components3)
# plot_top_words(lda5, tf_feature_names, 10, 'Распределение слов по темам, LDA-модель', n_components5)


#результаты максимально неинформативны

#попробуем кластеризовать тексты c разными векторайзерами
Tfidf = TfidfVectorizer()
count = CountVectorizer()
X = news_df['text_lemmas']
X_tf = Tfidf.fit_transform(X)
X_count = count.fit_transform(X)

k_means_clusters_3 = KMeans(n_clusters=3, random_state=0).fit(X_tf)
k_means_clusters_5 = KMeans(n_clusters=5, random_state=0).fit(X_tf)
k_means_clusters_7 = KMeans(n_clusters=7, random_state=0).fit(X_tf)
k_means_clusters_10 = KMeans(n_clusters=10, random_state=0).fit(X_tf)
print(silhouette_score(X_tf, k_means_clusters_3.labels_))
print(silhouette_score(X_tf, k_means_clusters_5.labels_))
print(silhouette_score(X_tf, k_means_clusters_7.labels_))
print(silhouette_score(X_tf, k_means_clusters_10.labels_))


k_means_clusters_3_с = KMeans(n_clusters=10, random_state=0).fit(X_count)
k_means_clusters_5_с = KMeans(n_clusters=5, random_state=0).fit(X_count)
k_means_clusters_7_с = KMeans(n_clusters=20, random_state=0).fit(X_count)
k_means_clusters_10_с = KMeans(n_clusters=3, random_state=0).fit(X_count)
# print(silhouette_score(X_count, k_means_clusters_3_с.labels_))
# print(silhouette_score(X_count, k_means_clusters_5_с.labels_))
# print(silhouette_score(X_count, k_means_clusters_7_с.labels_))
# print(silhouette_score(X_count, k_means_clusters_10_с.labels_))

#silhouette_score также очень низкий, не удалось нормально поделить новостную выборку на кластеры


