# Слова: «ветерок» и «дымок»

from gensim.models import KeyedVectors

# positive_list = ['дым_NOUN', 'ветер_NOUN'] Вариант для ленивых
positive_list = ['воздух_NOUN', 'легкий_ADJ', 'костерок_NOUN']
negative_list = []

word2vec_model = KeyedVectors.load_word2vec_format("cbow.txt", binary=False)
dist = word2vec_model.most_similar(positive=positive_list, negative=negative_list)

for i in dist:
    print(i)
