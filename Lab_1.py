import nltk

from pymorphy3 import MorphAnalyzer

nltk.download('punkt_tab')

morph_analyzer = MorphAnalyzer()

with open('University.txt', 'r', encoding='utf8') as file:
    text = file.read()

sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    for index in range(len(words) - 1):
        current_word = morph_analyzer.parse(words[index])[0]
        next_word = morph_analyzer.parse(words[index + 1])[0]
        if (
            (
            (current_word.tag.POS == "NOUN" and next_word.tag.POS == "ADJF") or
            (current_word.tag.POS == "ADJF" and next_word.tag.POS == "NOUN")
            ) and
            (current_word.tag.gender == next_word.tag.gender) and
            (current_word.tag.number == next_word.tag.number) and
            (current_word.tag.case == next_word.tag.case)
        ):
            print(current_word.normal_form + " " + next_word.normal_form)
