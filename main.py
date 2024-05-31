import telebot
from PyMultiDictionary import MultiDictionary, DICT_EDUCALINGO
import re
# import nltk
# nltk.download('punkt')
from ruts import ReadabilityStats
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

bot = telebot.TeleBot('7249611408:AAFEKAzWtAGG_20kqisF4XT-6Th7yhV7DDI')
save_dir = '/Users/rdashk/PycharmProjects/mtc/users_files'
new_filename = f'{save_dir}/test.txt'


@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    bot.send_message(message.chat.id, 'Бот готов! Жду ваш документ')


@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    global save_dir

    try:
        file_name = message.document.file_name
        file_id_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_id_info.file_path)

        new_filename = f'{save_dir}/{message.chat.id}_{file_name}'
        with open(new_filename, 'wb') as new_file:
            new_file.write(downloaded_file)

        text = ""
        arr = []
        with open(new_filename, 'r', encoding="ascii", errors="surrogateescape") as f:
            text = f.read()
            arr = text.split(".")

        preprocessed_text = preprocess_text(text)

        # текст на отдельные слова, удаление стоп-слов
        words = word_tokenize(preprocessed_text)
        stop_words = set(stopwords.words('russian'))
        filtered_words = [word for word in words if word not in stop_words]

        # TF-IDF для определения наиболее характерных слов
        stemmer = SnowballStemmer("russian")
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        stemmed_text = ' '.join(stemmed_words)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([stemmed_text])

        feature_names = tfidf_vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense()
        word_scores = dense.tolist()[0]

        # Сортировка слов по TF-IDF
        # word_scores_sorted = sorted(zip(range(len(word_scores)), word_scores), key=lambda x: x[1], reverse=True)

        # Из наиболее характерных слов берем подходящие по индексу
        new_filename = f'{save_dir}/new_{message.chat.id}_{file_name}'
        with open(new_filename, 'w') as new_file:
            for ar in arr:
                ine = set(word_scores) & set(ar.lower().split(" "))
                if len(ine) > 0:
                    rs = ReadabilityStats(text)
                    stats = rs.get_stats()

                    if stats['flesch_reading_easy'] < 30 and \
                        stats['coleman_liau_index'] > 10 and \
                        stats['smog_index'] > 8 and \
                        stats['automated_readability_index'] > 6:
                        for w in ine:
                            bot.send_message(message.chat.id, get_simple_definition(w))
                            new_file.write(ar + '.\n' + w +
                                                      '— это значит, что ' + get_simple_definition(w) + '.\n')
                        bot.send_message(message.chat.id, "Проверьте ваш файл и отправьте повторно")


    except Exception as ex:
        # bot.send_message(message.chat.id, "Проверьте ваш файл и отправьте повторно")
        bot.send_message(message.chat.id, ex)


def get_simple_definition(word: str) -> str:
    dictionary = MultiDictionary()
    definitions = dictionary.meaning('ru', word)
    mean = definitions[1]
    arr = []
    for m in mean.split(" "):
        nm = re.sub(r'[^а-яА-Я]', '', m)
        if len(nm)>2:
            arr.append(nm)
    return ' '.join(arr)


# обработка текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # без пунктуации
    return text


bot.polling(none_stop=True)