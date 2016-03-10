# -*- coding: utf-8 -*-

import re

from lxml import etree
from tqdm import tqdm

from falsefriends.wikiextractor import clean, section

RE_LINKS_FILES = re.compile(r'\[\[([\s\(\)\w\-]*):([\s\(\)\w\-]*)[\s]*?\|*?[\s\(\)\w\-]*?\]\]',
                            re.IGNORECASE | re.UNICODE | re.DOTALL)
RE_PUNCTUATION = re.compile(r'\W', re.IGNORECASE | re.UNICODE)

ALPHABET_EN = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z'}
ALPHABET = {
    'en': ALPHABET_EN,
    'es': ALPHABET_EN | {'á', 'é', 'í', 'ñ', 'ó', 'ú', 'ü'},
    'pt': ALPHABET_EN | {'ç', 'á', 'à', 'â', 'ã', 'é', 'è', 'í', 'ó', 'ô', 'õ', 'ú'},
}

ADMITED_ARTICLES_TYPES = {'0', # normal wikipedia articles
                          '4', # articles about wikipedia itself
                          '12', # help articles
                          '100', # wikipedia portals
                          '102' # wikiprojects
                          }


def valid_word(word_in_lowercase, lang):
    return all(letter in ALPHABET[lang] for letter in word_in_lowercase)


def paragraphs(wiki_text):
    return wiki_text.lstrip()


# noinspection SpellCheckingInspection
def replace_digits_with_words(text, lang):
    if lang == 'es':
        return text \
            .replace('0', ' cero ') \
            .replace('1', ' uno ') \
            .replace('2', ' dos ') \
            .replace('3', ' tres ') \
            .replace('4', ' cuatro ') \
            .replace('5', ' cinco ') \
            .replace('6', ' seis ') \
            .replace('7', ' siete ') \
            .replace('8', ' ocho ') \
            .replace('9', ' nueve ')
    elif lang == 'pt':
        return text \
            .replace('0', ' zero ') \
            .replace('1', ' um ') \
            .replace('2', ' dois ') \
            .replace('3', ' três ') \
            .replace('4', ' quatro ') \
            .replace('5', ' cinco ') \
            .replace('6', ' seis ') \
            .replace('7', ' sete ') \
            .replace('8', ' oito ') \
            .replace('9', ' nove ')
    else:
        raise ValueError("{} is not a valid language".format(lang))


def leave_only_letters(line, lang):
    line_without_digits = replace_digits_with_words(line, lang)
    words_without_punctuation_and_digits = RE_PUNCTUATION.sub(' ', line_without_digits).split()
    valid_words = [word for word in words_without_punctuation_and_digits if valid_word(word, lang)]
    # noinspection PyTypeChecker
    return ' '.join(valid_words)


def pre_process_wiki(input_file_name, output_file_name, lang):
    context = etree.iterparse(input_file_name, tag='page')
    with open(output_file_name, 'w') as output_file:
        if lang == 'es':
            if 'sample' in input_file_name:
                articles = 61
            else:
                articles = 1242337
        else:
            articles = 912133

        progress_bar = tqdm(total=articles)

        for _, page_elem in context:
            ns_elem = page_elem.find('ns')
            if ns_elem is not None and ns_elem.text.strip() in ADMITED_ARTICLES_TYPES:
                redirect_elem = page_elem.find('redirect')

                if redirect_elem is None:
                    text_elem = page_elem.find('revision/text')
                    text = text_elem.text
                    if text is not None:
                        text = RE_LINKS_FILES.sub('', text)
                        text = paragraphs(text)
                        text = clean(text)
                        text = section.sub('', text)
                        text = text.lower()
                        text = '\n'.join(
                            line for line in (
                                leave_only_letters(line, lang) for line in text.split('\n')
                            ) if line != ''
                        )
                        output_file.write(text + '\n')

                    text_elem.clear()
                else:
                    redirect_elem.clear()

            ns_elem.clear()
            page_elem.clear()
            while page_elem.getprevious() is not None:
                del page_elem.getparent()[0]

            progress_bar.update()

    del context
