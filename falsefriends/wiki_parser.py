# -*- coding: utf-8 -*-

import re

from lxml import etree
from tqdm import tqdm

from falsefriends.wikiextractor import clean, section

RE_LINKS_FILES = re.compile(r'\[\[([\s\(\)\w\-]*):([\s\(\)\w\-]*)[\s]*?\|*?[\s\(\)\w\-]*?\]\]',
                            re.IGNORECASE | re.UNICODE | re.DOTALL)
RE_PUNCTUATION = re.compile(r'\W', re.IGNORECASE | re.UNICODE)


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


def remove_non_letters(text, lang):
    return ' '.join(RE_PUNCTUATION.sub(' ', replace_digits_with_words(text, lang)).split())


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
            redirect_elem = page_elem.find('redirect')

            if redirect_elem is None:
                text_elem = page_elem.find('revision/text')
                text = text_elem.text

                if text is not None:
                    text = RE_LINKS_FILES.sub('', text)
                    text = paragraphs(text)
                    text = clean(text)
                    text = section.sub('', text)
                    text = '\n'.join(
                        line for line in (
                            remove_non_letters(line, lang) for line in text.split('\n')
                        ) if line != ''
                    )
                    text = text.lower()
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
