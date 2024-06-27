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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
#spacy.cli.download('ru_core_news_sm') #закомментировано, чтоб каждый раз не скачивать модель

stop_words = stopwords.words('russian')


#парсинг новостей общей тематики
# url = "https://www.mk.ru/news/2024/6/9/"
# page = rq.get(url)
# soup = BeautifulSoup(page.text, features="html.parser")
#
# news_title = soup.find_all('h3', {'class' : 'news-listing__item-title'})
# text_title = [i.text for i in news_title]
#
#
# links = []
# for i in soup.find_all('a', {'class' : 'news-listing__item-link'}):
#       links.append(i.get('href'))
#
#
# def GetNewsBody(url):
#   page = rq.get(url)
#   soup = BeautifulSoup(page.text, features="html.parser")
#   body_text = []
#   for item in soup.find_all('div', {'class': 'article__body'}):
#         for i in item.find_all('p'):
#             body_text.append(i.text.strip())
#
#   return ' '.join(body_text)
#
# news_body = []
# for link in links:
#     new = GetNewsBody(link)
#     news_body.append(new)
#
#
# for i in news_body:
#     i = i.replace('\xa0', ' ')
#
#
# # print(news_body)
# #
# news_df = pd.DataFrame({'link': links, 'title': text_title, 'text': news_body})
# # news_df.to_csv('newsMK_df.csv')
# # print(news_df)

nlp_rus = spacy.load("ru_core_news_sm")
# #сделаем отдельный столбец с очищенным и лемматизированным текстом
# def preprocess(text):
#     tokenized = word_tokenize(text)
#     text_clean = []
#     for word in tokenized:
#         if word[0].isalnum() and word not in stop_words:
#             text_clean.append(word)
#     doc = nlp_rus(' '.join(text_clean)) #передаем в spacy и лемматизируем
#     lemmas = []
#     for token in doc:
#         lemmas.append(token.lemma_)
#     return ' '.join(lemmas)
#
# news_df['text_lemmas'] = news_df['text'].apply(preprocess)

#закоментировала строки выше, чтобы не парсить каждый раз, иначе там срабатывает firewall, загрузила полученный ранее датасет из файла
news_df = pd.read_csv('newsMK_df.csv')

# #выделение именнованных сущностей для каждой статьи

ents = [] #список списков для всех NER по статьям
for i in news_df['text_lemmas']:
    doc = nlp_rus(i)
    article_ents = [] #список для NER одной статьи
    for ent in doc.ents:
        article_ents.append((ent.text, ent.label_))
    ents.append(article_ents)

news_df['NER'] = ents #добавляем колонку с NER в датафрейм

#то что спейси не определил как NER или не нашел
FACs = ['atacms', 'миг-29', 'су-27', 'су-25', 'Су-24', 'covid-19', 'истребитель Mirage 2000-5']
LOCs = ['улица сарьяна', 'улица прошяну', 'азербайджанская ССР', 'сектор газа', 'река иордан', 'киев', 'уимбилдон', 'хонсю', 'южный осетия', 'беверли-хиллз', 'амхерст', 'цхинвальском регионе', 'миргород', 'поселок прибрежный', 'обь', 'регион пил', 'единая осетия', 'поселок аэропорт', 'миллионный улица', 'мексиканская республика', 'люберцах']
PERs = ['гарник даниелян', 'биньямин нетаньяху', 'джо байден', 'хейсканен', 'кафельников', 'бахыш бахышлы', 'олаф шольц', 'теодор постол', 'аскеров', 'ныхас', 'иры фарн', 'шломи зив', 'ноа аргамани', 'данил миленин', 'фарида']
ORGs = ['мхат', 'олимпийский игра', 'армянская апостольская церковь', 'РИА Новости', 'Sky News', 'ЦАХАЛ', 'YouGov', 'the national', 'Ferrari Challenge Japan', 'бундесвер', 'тасс', 'зов народ', 'пмэф', 'федерация спортивный борьба', 'говорит москва', 'белый дом', 'файтбомбер', 'егэ', 'министерство здравоохранения', 'эрмитаж', 'миницифры', 'портал госуслуг', 'канал известие', 'йеменские хуситы']
def add_ner(text):
    additional_ners = []
    for word in FACs:
        if word in text:
            a = (word, 'FAC')
            additional_ners.append(a)
    for word in LOCs:
        if word in text:
            a = (word, 'LOC')
            additional_ners.append(a)
    for word in PERs:
        if word in text:
            a = (word, 'PER')
            additional_ners.append(a)
    for word in ORGs:
        if word in text:
            a = (word, 'ORG')
            additional_ners.append(a)

    return additional_ners

#смотрим есть ли пропущенные spacy NER в лемматизированном тексте и добавляем столбец с ними в датафрейм
news_df['additional_ners'] = news_df['text_lemmas'].apply(add_ner)

#соединяем NER полученные spacy и то^ xnj spacy пропустил для каждой статьи, чтобы был полный список хештегов
additional_ners = news_df['additional_ners'].tolist()
combined_list_ner = [a + b for a, b in zip(ents, additional_ners)]

hashtags = []
hashtags_final = []
for kort in combined_list_ner:
    all_hash = []
    for i in kort:
        if i[1] == 'LOC' or i[1] == 'PER': #делаем NER c лейблами PER и LOC с заглавной буквы для имен и стран или все буквы заглавные для аббревиатур(длина строки меньше или равно3)
           if len(i[0]) <= 3:
               ner = i[0].upper()
           else:
               ner = i[0].title()
        else:
            ner = i[0]
        all_hash.append(ner) #сохраняем все NER без лейбла
        prom = []
        for k in all_hash:
            k = re.sub(' - ', '-', k) #убираем лишние пробелы рядом с дефисами, которые появились после лемматизации
            k = re.sub(' ', '_', k) #склеиваем NER где больше 1 слова нижним подчеркиванием
            prom.append(k)
        uniq_hash = set(prom) #убираем повторы NER для каждой строки
    hashtags.append(uniq_hash)
    for h in hashtags: #добавляем решетку, убираем формат списка
        s = ''
        for j in h:
            s += ' #' + j
    hashtags_final.append(s)

news_df['hashtags'] = hashtags_final


news_df.to_csv('newsMK_df.csv', index=False)


#предобработанный текст делим на тренировочную и тестовую выборку, чтобы иметь возможность оценить качество работы модели
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

plot_top_words(lda2, tf_feature_names, 10, 'Распределение слов по темам, LDA-модель', n_components2)
plot_top_words(lda3, tf_feature_names, 10, 'Распределение слов по темам, LDA-модель', n_components3)
plot_top_words(lda5, tf_feature_names, 10, 'Распределение слов по темам, LDA-модель', n_components5)


#результаты максимально неинформативны

#попробуем кластеризовать все тексты корпуса c разными векторайзерами с помощью KMeans
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
print(silhouette_score(X_count, k_means_clusters_3_с.labels_))
print(silhouette_score(X_count, k_means_clusters_5_с.labels_))
print(silhouette_score(X_count, k_means_clusters_7_с.labels_))
print(silhouette_score(X_count, k_means_clusters_10_с.labels_))

#silhouette_score также очень низкий, не удалось нормально поделить новостную выборку на кластеры

#попробуем поделить на кластеры с помощью DBSCAN
clustering = DBSCAN(eps=1, min_samples=40).fit(X_tf)
print(set(clustering.labels_))
clustering_2 = DBSCAN(eps=0.2, min_samples=2).fit(X_tf)
print(set(clustering_2.labels_))
clustering_3 = DBSCAN(eps=0.5, min_samples=3).fit(X_tf)
print(set(clustering_3.labels_))
clustering_4 = DBSCAN(eps=1, min_samples=5).fit(X_tf)
print(set(clustering_4.labels_))
clustering_5 = DBSCAN(eps=2, min_samples=10).fit(X_tf)
print(set(clustering_5.labels_))
clustering_6 = DBSCAN(eps=3, min_samples=15).fit(X_tf)
print(set(clustering_6.labels_))
clustering_7 = DBSCAN(eps=5, min_samples=10).fit(X_tf)
print(set(clustering_7.labels_))

clustering8 = DBSCAN(eps=1, min_samples=40).fit(X_count)
print(set(clustering8.labels_))
clustering9 = DBSCAN(eps=0.2, min_samples=2).fit(X_count)
print(set(clustering9.labels_))
clustering10 = DBSCAN(eps=0.5, min_samples=3).fit(X_count)
print(set(clustering10.labels_))
clustering11 = DBSCAN(eps=1, min_samples=5).fit(X_count)
print(set(clustering11.labels_))
clustering12 = DBSCAN(eps=2, min_samples=10).fit(X_count)
print(set(clustering12.labels_))
clustering13 = DBSCAN(eps=3, min_samples=15).fit(X_count)
print(set(clustering13.labels_))
clustering14 = DBSCAN(eps=5, min_samples=10).fit(X_count)
print(set(clustering14.labels_))
clustering15 = DBSCAN(eps=0.1, min_samples=3).fit(X_count)
print(set(clustering15.labels_))
clustering16 = DBSCAN(eps=0.1, min_samples=2).fit(X_count)
print(set(clustering16.labels_))

#DBSCAN находит только 1 кластер 0 или только шум -1, silhouette_score в этом случае не сработает
