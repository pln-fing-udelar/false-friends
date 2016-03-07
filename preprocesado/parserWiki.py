#!/usr/bin/python
# -*- coding: utf-8 -*-
from cgi import escape
import csv
import os
import re
from lxml import etree
from WikiExtractor import clean, dropNested, section
from nltk.corpus import stopwords
from nltk.tokenize import SpaceTokenizer
import datetime
from multiprocessing import Process


def parrafos(texto_wiki):
    resultado = texto_wiki.lstrip()
    return resultado


def preprocesar_wiki(infile, outfile, lang):

    def replace_numbers_with_words(s):
        if lang == 'es':
            return s.replace('0', u' cero ')\
            .replace('1', u' uno ')\
            .replace('2', u' dos ')\
            .replace('3', u' tres ')\
            .replace('4', u' cuatro ')\
            .replace('5', u' cinco ')\
            .replace('6', u' seis' )\
            .replace('7', u' siete ')\
            .replace('8', u' ocho ')\
            .replace('9', u' nueve ')
        elif lang == 'pt':
            return s.replace('0', ' zero ')\
            .replace('1', u' um ')\
            .replace('2', u' dois ')\
            .replace('3', u' três ')\
            .replace('4', u' quatro ')\
            .replace('5', u' cinco ')\
            .replace('6', u' seis' )\
            .replace('7', u' sete ')\
            .replace('8', u' oito ')\
            .replace('9', u' nove ')

    context = etree.iterparse(infile, events=('end',), tag='page', encoding='utf-8')
    print datetime.datetime.now()

    links_archivos = re.compile("\[\[([\s\(\)\w\-]*)\:([\s\(\)\w\-]*)[\s]*?\|*?[\s\(\)\w\-]*?\]\]".decode('utf-8'),
                                re.IGNORECASE | re.UNICODE | re.DOTALL)
    remove_punctuation_re = re.compile("\W", re.IGNORECASE | re.UNICODE)

    def remove_non_letter(item):
        return ' '.join(remove_punctuation_re.sub(' ', replace_numbers_with_words(item)).split())

    f = open(outfile, 'w')
    # por cada elemento del artículo de wikipedia
    for event, elem in context:
        ns = elem.find("ns")
        redirect = elem.find("redirect")
        # hay un redirect
        if redirect is not None:
            redirect.clear()
        # no es  un redirect
        else:
            # obtengo el titulo y el texto
            titulo = elem.find("title")
            texto = elem.find("revision/text")
            texto_text = unicode(texto.text)
            titulo_str = unicode(titulo.text).lower()
            texto_text = dropNested(texto_text, r'{{', r'}}')
            texto_text = dropNested(texto_text, r'{\|', r'\|}')
            texto_text = links_archivos.sub('', texto_text)
            texto_text = parrafos(texto_text)
            texto_text = clean(texto_text)
            texto_text = section.sub('', texto_text)
            texto_text = '\n'.join(filter(lambda item: item != '', map(remove_non_letter, texto_text.split('\n'))))
            texto_text = texto_text.lower()
            f.write(texto_text.encode('utf-8') + '\n')
            titulo.clear()
            texto.clear()
        ns.clear()
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    f.close()
    del context
    print datetime.datetime.now()

if __name__ == '__main__':
    p = Process(target=preprocesar_wiki, args=('wikipedia_es.xml', 'wikipedia_es_preprocesada.txt', 'es'))
    p.start()
    preprocesar_wiki('wikipedia_pt.xml', 'wikipedia_pt_preprocesada.txt', 'pt')
    p.join()
# preprocesar_wiki('prueba.xml', 'prueba_preprocesada.txt', 'es')
