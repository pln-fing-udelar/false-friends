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


stopwords_set = set([unicode(palabra) for palabra in stopwords.words('spanish')])


def descartable(s):
    try:
        float(s)
        return True
    except ValueError:
        if len(s) < 3 or len(s) > 60:
            return True
        else:
            return s in stopwords_set


def parrafos(texto_wiki):
    resultado = texto_wiki.lstrip()
    return resultado


def replace_numbers_with_words(s):
    return s.replace('0', ' cero ')\
        .replace('1', ' uno ')\
        .replace('2', ' dos ')\
        .replace('3', ' tres ')\
        .replace('4', ' cuatro ')\
        .replace('5', ' cinco ')\
        .replace('6', ' seis' )\
        .replace('7', ' siete ')\
        .replace('8', ' ocho ')\
        .replace('9', ' nueve ')\


def preprocesar_wiki(infile):
    identificador = 0
    infile_xml = infile + ".xml"
    context = etree.iterparse(infile_xml, events=('end',), tag='page', encoding='utf-8')
    print datetime.datetime.now()
    # desam_regex = re.compile("\{\{desambiguaci[oó]n\}\}".decode('utf-8'), re.IGNORECASE | re.UNICODE | re.DOTALL)
    # categ_regex = re.compile("\[\[Categor[ií]a\:([\s\(\)\w\-]*)[\s]*?\|?[\s\(\)\w\-]*?\]\]".decode('utf-8'),
    #                          re.IGNORECASE | re.UNICODE | re.DOTALL)
    links_archivos = re.compile("\[\[([\s\(\)\w\-]*)\:([\s\(\)\w\-]*)[\s]*?\|*?[\s\(\)\w\-]*?\]\]".decode('utf-8'),
                                re.IGNORECASE | re.UNICODE | re.DOTALL)
    remove_punctuation_re = re.compile("\W", re.IGNORECASE | re.UNICODE)

    def remove_non_letter(item):
        return ' '.join(remove_punctuation_re.sub(' ', replace_numbers_with_words(item)).split())

    f = open(infile + '_preprocesada.txt', 'w')
    # por cada elemento del artículo de wikipedia
    for event, elem in context:
        ns = elem.find("ns")
        redirect = elem.find("redirect")
        # hay un redirect y no es una pagina de wikipedia especial
        if redirect is not None:
            redirect.clear()
        # no es una pagina de wikipedia especial ni tampoco un redirect
        # elif ns.text == "0":
        else:
            # obtengo el titulo y el texto
            titulo = elem.find("title")
            texto = elem.find("revision/text")
            texto_text = unicode(texto.text)
            titulo_str = unicode(titulo.text).lower()
            # lo necesito afuera por si esta pertenece a desambiguacion, extraigo categorias
            # categorias_art = map(unicode, categ_regex.findall(texto_text))
            # me fijo si la categoria desambiguacion no pertenece al articulo, se cumple para todos que es none
            # todos estor chequeos son para saber si no es una pagina de desambiguacion
            # bandera_desambiguacion = 'desambiguación'.decode('utf-8') not in categorias_art
            # bandera_desambiguacion &= 'desambiguación'.decode('utf-8') not in categorias_art
            # bandera_desambiguacion &= 'Desambiguación'.decode('utf-8') not in categorias_art
            # bandera_desambiguacion &= 'Desambiguación'.decode('utf-8') not in categorias_art
            # bandera_desambiguacion &= desam_regex.search(texto_text) is None
            # no es pagina de desambiguacion
            # if bandera_desambiguacion and not descartable(titulo_str):
            texto_text = dropNested(texto_text, r'{{', r'}}')
            texto_text = dropNested(texto_text, r'{\|', r'\|}')
            texto_text = links_archivos.sub('', texto_text)
            texto_text = parrafos(texto_text)
            texto_text = clean(texto_text)
            texto_text = section.sub('', texto_text)
            texto_text = texto_text.replace('*', '')
            texto_text = '\n'.join(filter(lambda item: item != '', map(remove_non_letter, texto_text.split('\n'))))
            f.write(texto_text.encode('utf-8'))
            ### hasta aca venia el if
            titulo.clear()
            texto.clear()
        ns.clear()
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    f.close()
    del context
    print datetime.datetime.now()

preprocesar_wiki('prueba')
