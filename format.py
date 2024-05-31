from gensim.models import LdaMulticore, Phrases
from gensim import corpora
from gensim import models


def get_n_gram(data, n = 2):

    # threshold - пороговая оценка для формирования фраз. Более высокие значения приводят к меньшему количеству фраз.
    new_data_pr = []
    bigram = Phrases(data, min_count=3, threshold=5.0, delimiter=' ')

    for idx in range(len(data)):
        new_data_pr.append([])
        # print("------------------new!-------------------")
        for token in bigram[data[idx]]:

            if ' ' in token:  # Token is a bigram, add to document.
                # print(token)
                new_data_pr[idx].append(token)
    # print("new=", new_data_pr)

    if n == 3:
        # print("old=", data_processed)
        trigram = models.phrases.Phrases(bigram[data], min_count=3, threshold=2.0, delimiter=' ')
        for idx in range(len(data)):
            new_data_pr.append([])
            # print("------------------new!-------------------")
            for token in trigram[bigram[data[idx]]]:

                if ' ' in token and token not in new_data_pr[idx]:
                    # print(token)
                    new_data_pr[idx].append(token)

    for idx in range(len(new_data_pr)):
        for phrase in new_data_pr[idx]:
            data[idx].append(phrase)

    answer = []
    set_data = []
    for se in data:
        set_data.append(set(se))

    return answer, set_data




'''
-------------------------------------------------------------
# обновление словаря
# documents_2 = ["The intersection graph of paths in trees",
#                "Graph minors IV Widths of trees and well quasi ordering",
#                "Graph minors A survey"]
# texts_2 = [[text for text in doc.split()] for doc in documents_2]
# dictionary.add_documents(texts_2)
-------------------------------------------------------------
'''

# Create dictionary
dictionary = corpora.Dictionary(data_pr)
dictionary.filter_extremes(no_below=1, no_above=0.99, keep_n=None) # отфильтруем словарь (удалим слишком редко
# или слишком часто встречающиеся слова)

'''
# сохранить словарь в файл, далее считать словарь из файла 
dct_filename = "dct_words"
dictionary.save_as_text(dct_filename)
loaded_dct = corpora.Dictionary.load_from_text(dct_filename)
'''


'''
корпус слов (“Мешок Слов“). Все объекты корпуса содержит id слова и его частоту в каждом документе.
'''
corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in data_pr]
# print(corpus)


'''
TF-IDF model
чем чаще слово встречается в тексте, тем меньше его вес
(слова, встречающиеся в документах чаще, приобретают меньший вес)
Слова, встречающиеся во всех документах, полностью исключаются

Term frequency weighing:
        b - binary,
        t or n - raw,
        a - augmented,
        l - logarithm,
        d - double logarithm,
        L - log average.

Document frequency weighting:
        x or n - none,
        f - idf,
        t - zero-corrected idf,
        p - probabilistic idf.

Document normalization:
        x or n - none,
        c - cosine,
        u - pivoted unique,
        b - pivoted character length.
'''

