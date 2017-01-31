#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# =============================================================================
#  Version: 2.6 (Oct 14, 2013)
#  Author: Giuseppe Attardi (attardi@di.unipi.it), University of Pisa
#          Antonio Fuschetto (fuschett@di.unipi.it), University of Pisa
#
#  Contributors:
#    Leonardo Souza (lsouza@amtera.com.br)
#    Juan Manuel Caicedo (juan@cavorite.com)
#    Humberto Pereira (begini@gmail.com)
#    Siegfried-A. Gevatter (siegfried@gevatter.com)
#    Pedro Assis (pedroh2306@gmail.com)
#
# =============================================================================
#  Copyright (c) 2009. Giuseppe Attardi (attardi@di.unipi.it).
# =============================================================================
#  This file is part of Tanl.
#
#  Tanl is free software; you can redistribute it and/or modify it
#  under the terms of the GNU General Public License, version 3,
#  as published by the Free Software Foundation.
#
#  Tanl is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""Wikipedia Extractor:
Extracts and cleans text from Wikipedia database dump and stores output in a
number of files of similar size in a given directory.
Each file contains several documents in Tanl document format:
    <doc id="" url="" title="">
        ...
    </doc>

Usage:
  WikiExtractor.py [options]

Options:
  -c, --compress        : compress output files using bzip
  -b, --bytes= n[KM]    : put specified bytes per output file (default 500K)
  -B, --base= URL       : base URL for the Wikipedia pages
  -l, --link            : preserve links
  -n NS, --ns NS        : accepted namespaces (separated by commas)
  -o, --output= dir     : place output files in specified directory (default
                          current)
  -s, --sections	: preserve sections
  -h, --help            : display this help and exit
"""

import bz2
import getopt
import os.path
import re
import sys
from html.entities import name2codepoint

# PARAMS ####################################################################

# This is obtained from the dump itself
prefix = None

##
# Whether to preserve links in output
#
keep_links = False

##
# Whether to transform sections into HTML
#
keep_sections = False

##
# Recognize only these namespaces
# w: Internal links to the Wikipedia
# wiktionary: Wiki dictionary
# wikt: shortcut for Wiktionary
#
accepted_namespaces = {'w', 'wiktionary', 'wikt'}

##
# Drop these elements from article text
#
discard_elements = {'gallery', 'timeline', 'noinclude', 'pre', 'table', 'tr', 'td', 'th', 'caption', 'form', 'input',
                    'select', 'option', 'textarea', 'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'menu', 'dir', 'ref',
                    'references', 'img', 'imagemap', 'source'}

# =========================================================================
#
# MediaWiki Markup Grammar

# Template = "{{" [ "msg:" | "msgnw:" ] PageName { "|" [ ParameterName "=" AnyText | AnyText ] } "}}" ;
# Extension = "<" ? extension ? ">" AnyText "</" ? extension ? ">" ;
# NoWiki = "<nowiki />" | "<nowiki>" ( InlineText | BlockText ) "</nowiki>" ;
# Parameter = "{{{" ParameterName { Parameter } [ "|" { AnyText | Parameter } ] "}}}" ;
# Comment = "<!--" InlineText "-->" | "<!--" BlockText "//-->" ;
#
# ParameterName = ? uppercase, lowercase, numbers, no spaces, some special chars ? ;
#
# ===========================================================================

# Program version
version = '2.5'


# Main function ###########################################################


def wiki_document(out, id_, title, text):
    url = get_url(id_, prefix)
    header = '<doc id="%s" url="%s" title="%s">\n' % (id_, url, title)
    # Separate header from text with a newline.
    header += title + '\n'
    header = header.encode('utf-8')
    text = clean(text)
    footer = "\n</doc>"
    out.reserve(len(header) + len(text) + len(footer))
    print(header, file=out)
    for line in compact(text):
        print(line.encode('utf-8'), file=out)
    print(footer, file=out)


def get_url(id_, prefix_):
    return "%s?curid=%s" % (prefix_, id_)


# ------------------------------------------------------------------------------

self_closing_tags = ['br', 'hr', 'nobr', 'ref', 'references']

# handle 'a' separately, depending on keepLinks
ignored_tags = [
    'b', 'big', 'blockquote', 'center', 'cite', 'div', 'em',
    'font', 'h1', 'h2', 'h3', 'h4', 'hiero', 'i', 'kbd', 'nowiki',
    'p', 'plaintext', 's', 'small', 'span', 'strike', 'strong',
    'sub', 'sup', 'tt', 'u', 'var',
]

placeholder_tags = {'math': 'formula', 'code': 'codice'}


def normalize_title(title):
    # remove leading whitespace and underscores
    title = title.strip(' _')
    # replace sequences of whitespace and underscore chars with a single space
    title = re.compile(r'[\s_]+').sub(' ', title)

    m = re.compile(r'([^:]*):(\s*)(\S(?:.*))').match(title)
    if m:
        prefix_ = m.group(1)
        if m.group(2):
            optional_whitespace = ' '
        else:
            optional_whitespace = ''
        rest = m.group(3)

        ns = prefix_.capitalize()
        if ns in accepted_namespaces:
            # If the prefix designates a known namespace, then it might be
            # followed by optional whitespace that should be removed to get
            # the canonical page name
            # (e.g., "Category:  Births" should become "Category:Births").
            title = ns + ":" + rest.capitalize()
        else:
            # No namespace, just capitalize first letter.
            # If the part before the colon is not a known namespace, then we must
            # not remove the space after the colon (if any), e.g.,
            # "3001: The_Final_Odyssey" != "3001:The_Final_Odyssey".
            # However, to get the canonical page name we must contract multiple
            # spaces into one, because
            # "3001:   The_Final_Odyssey" != "3001: The_Final_Odyssey".
            title = prefix_.capitalize() + ":" + optional_whitespace + rest
    else:
        # no namespace, just capitalize first letter
        title = title.capitalize()
    return title


##
# Removes HTML or XML character references and entities from a text string.
#
# @param text The HTML (or XML) source text.
# @return The plain text, as a Unicode string, if necessary.

def unescape(text):
    def fix_up(m):
        text_ = m.group(0)
        code = m.group(1)
        try:
            if text_[1] == "#":  # character reference
                if text_[2] == "x":
                    return chr(int(code[1:], 16))
                else:
                    return chr(int(code))
            else:  # named entity
                return chr(name2codepoint[code])
        except (KeyError, ValueError):
            return text_  # leave as is

    return re.sub("&#?(\w+);", fix_up, text)


# Match HTML comments
comment = re.compile(r'<!--.*?-->', re.DOTALL)

# Match elements to ignore
discard_element_patterns = []
for tag in discard_elements:
    pattern = re.compile(r'<\s*%s\b[^>]*>.*?<\s*/\s*%s>' % (tag, tag), re.DOTALL | re.IGNORECASE)
    discard_element_patterns.append(pattern)

# Match ignored tags
ignored_tag_patterns = []


def ignore_tag(tag_):
    left = re.compile(r'<\s*%s\b[^>]*>' % tag_, re.IGNORECASE)
    right = re.compile(r'<\s*/\s*%s>' % tag_, re.IGNORECASE)
    ignored_tag_patterns.append((left, right))


for tag in ignored_tags:
    ignore_tag(tag)

# Match selfClosing HTML tags
self_closing_tag_patterns = []
for tag in self_closing_tags:
    pattern = re.compile(r'<\s*%s\b[^/]*/\s*>' % tag, re.DOTALL | re.IGNORECASE)
    self_closing_tag_patterns.append(pattern)

# Match HTML placeholder tags
placeholder_tag_patterns = []
for tag, replacement in placeholder_tags.items():
    pattern = re.compile(r'<\s*%s(\s*| [^>]+?)>.*?<\s*/\s*%s\s*>' % (tag, tag), re.DOTALL | re.IGNORECASE)
    placeholder_tag_patterns.append((pattern, replacement))

# Match pre-formatted lines
pre_formatted = re.compile(r'^ .*?$', re.MULTILINE)

# Match external links (space separates second optional parameter)
external_link = re.compile(r'\[\w+.*? (.*?)\]')
external_link_no_anchor = re.compile(r'\[\w+[&\]]*\]')

# Matches bold/italic
bold_italic = re.compile(r"'''''([^']*?)'''''")
bold = re.compile(r"'''(.*?)'''")
italic_quote = re.compile(r"''\"(.*?)\"''")
italic = re.compile(r"''([^']*)''")
quote_quote = re.compile(r'""(.*?)""')

# Matches space
spaces = re.compile(r' {2,}')

# Matches dots
dots = re.compile(r'\.{4,}')


# A matching function for nested expressions, e.g. namespaces and tables.
def drop_nested(text, open_delimiter, close_delimiter):
    open_re = re.compile(open_delimiter)
    close_re = re.compile(close_delimiter)
    # partition text in separate blocks { } { }
    matches = []  # pairs (s, e) for each partition
    nest = 0  # nesting level
    start = open_re.search(text, 0)
    if not start:
        return text
    end = close_re.search(text, start.end())
    next_ = start
    while end:
        next_ = open_re.search(text, next_.end())
        if not next_:  # termination
            while nest:  # close all pending
                nest -= 1
                end0 = close_re.search(text, end.end())
                if end0:
                    end = end0
                else:
                    break
            matches.append((start.start(), end.end()))
            break
        while end.end() < next_.start():
            # { } {
            if nest:
                nest -= 1
                # try closing more
                last = end.end()
                end = close_re.search(text, end.end())
                if not end:  # unbalanced
                    if matches:
                        span = (matches[0][0], last)
                    else:
                        span = (start.start(), last)
                    matches = [span]
                    break
            else:
                matches.append((start.start(), end.end()))
                # advance start, find next close
                start = next_
                end = close_re.search(text, next_.end())
                break  # { }
        if next_ != start:
            # { { }
            nest += 1
    # collect text outside partitions
    res = ''
    start = 0
    for s, e in matches:
        res += text[start:s]
        start = e
    res += text[start:]
    return res


def drop_spans(matches, text):
    """Drop from text the blocks identified in matches
    :param text:
    :param matches:
    """
    matches.sort()
    res = ''
    start = 0
    for s, e in matches:
        res += text[start:s]
        start = e
    res += text[start:]
    return res


# Match interwiki links, | separates parameters.
# First parameter is displayed, also trailing concatenated text included
# in display, e.g. s for plural).
#
# Can be nested [[File:..|..[[..]]..|..]], [[Category:...]], etc.
# We first expand inner ones, than remove enclosing ones.
#
wiki_link = re.compile(r'\[\[([^[]*?)(?:\|([^[]*?))?\]\](\w*)')

parametrized_link = re.compile(r'\[\[.*?\]\]')


# Function applied to wikiLinks
def make_anchor_tag(match):
    global keep_links
    link = match.group(1)
    colon = link.find(':')
    if colon > 0 and link[:colon] not in accepted_namespaces:
        return ''
    trail = match.group(3)
    anchor = match.group(2)
    if not anchor:
        anchor = link
    anchor += trail
    if keep_links:
        return '<a href="%s">%s</a>' % (link, anchor)
    else:
        return anchor


def clean(text):
    # FIXME: templates should be expanded
    # Drop transclusions (template, parser functions)
    # See: http://www.mediawiki.org/wiki/Help:Templates
    text = drop_nested(text, r'{{', r'}}')

    # Drop tables
    text = drop_nested(text, r'{\|', r'\|}')

    # Expand links
    text = wiki_link.sub(make_anchor_tag, text)
    # Drop all remaining ones
    text = parametrized_link.sub('', text)

    # Handle external links
    text = external_link.sub(r'\1', text)
    text = external_link_no_anchor.sub('', text)

    # Handle bold/italic/quote
    text = bold_italic.sub(r'\1', text)
    text = bold.sub(r'\1', text)
    text = italic_quote.sub(r'&quot;\1&quot;', text)
    text = italic.sub(r'&quot;\1&quot;', text)
    text = quote_quote.sub(r'\1', text)
    text = text.replace("'''", '').replace("''", '&quot;')

    # Process HTML ###############

    # turn into HTML
    text = unescape(text)
    # do it again (&amp;nbsp;)
    text = unescape(text)

    # Collect spans

    matches = []
    # Drop HTML comments
    for m in comment.finditer(text):
        matches.append((m.start(), m.end()))

    # Drop self-closing tags
    for pattern_ in self_closing_tag_patterns:
        for m in pattern_.finditer(text):
            matches.append((m.start(), m.end()))

    # Drop ignored tags
    for left, right in ignored_tag_patterns:
        for m in left.finditer(text):
            matches.append((m.start(), m.end()))
        for m in right.finditer(text):
            matches.append((m.start(), m.end()))

    # Bulk remove all spans
    text = drop_spans(matches, text)

    # Cannot use dropSpan on these since they may be nested
    # Drop discarded elements
    for pattern_ in discard_element_patterns:
        text = pattern_.sub('', text)

    # Expand placeholders
    for pattern_, placeholder in placeholder_tag_patterns:
        index = 1
        for match in pattern_.finditer(text):
            text = text.replace(match.group(), '%s_%d' % (placeholder, index))
            index += 1

    text = text.replace('<<', 'Â«').replace('>>', 'Â»')

    #############################################

    # Drop pre-formatted
    # This can't be done before since it may remove tags
    text = pre_formatted.sub('', text)

    # Cleanup text
    text = text.replace('\t', ' ')
    text = spaces.sub(' ', text)
    text = dots.sub('...', text)
    text = re.sub(' (,:\.\)\]Â»)', r'\1', text)
    text = re.sub('(\[\(Â«) ', r'\1', text)
    text = re.sub(r'\n\W+?\n', '\n', text)  # lines with only punctuations
    text = text.replace(',,', ',').replace(',.', '.')
    return text


section = re.compile(r'(==+)\s*(.*?)\s*\1')


def compact(text):
    """Deal with headers, lists, empty sections, residuals of tables
    :param text:
    """
    page = []  # list of paragraph
    headers = {}  # Headers for unfilled sections
    empty_section = False  # empty sections are discarded

    for line in text.split('\n'):
        if not line:
            continue
        # Handle section titles
        m = section.match(line)
        if m:
            title = m.group(2)
            lev = len(m.group(1))
            if keep_sections:
                page.append("<h%d>%s</h%d>" % (lev, title, lev))
            if title and title[-1] not in '!?':
                title += '.'
            headers[lev] = title
            # drop previous headers
            for i in list(headers.keys()):
                if i > lev:
                    del headers[i]
            empty_section = True
            continue
        # Handle page title
        if line.startswith('++'):
            title = line[2:-2]
            if title:
                if title[-1] not in '!?':
                    title += '.'
                page.append(title)
        # handle lists
        elif line[0] in '*#:;':
            if keep_sections:
                page.append("<li>%s</li>" % line[1:])
            else:
                continue
        # Drop residuals of lists
        elif line[0] in '{|' or line[-1] in '}':
            continue
        # Drop irrelevant lines
        elif (line[0] == '(' and line[-1] == ')') or line.strip('.-') == '':
            continue
        elif len(headers):
            items = list(headers.items())
            items.sort()
            for (i, v) in items:
                page.append(v)
            headers.clear()
            page.append(line)  # first line
            empty_section = False
        elif not empty_section:
            page.append(line)

    return page


def handle_unicode(entity):
    numeric_code = int(entity[2:-1])
    if numeric_code >= 0x10000:
        return ''
    return chr(numeric_code)


# ------------------------------------------------------------------------------

class OutputSplitter:
    def __init__(self, compress, max_file_size, path_name):
        self.dir_index = 0
        self.file_index = -1
        self.compress = compress
        self.max_file_size = max_file_size
        self.path_name = path_name
        self.out_file = self.open_next_file()

    def reserve(self, size):
        cur_file_size = self.out_file.tell()
        if cur_file_size + size > self.max_file_size:
            self.close()
            self.out_file = self.open_next_file()

    def write(self, text):
        self.out_file.write(text)

    def close(self):
        self.out_file.close()

    def open_next_file(self):
        self.file_index += 1
        if self.file_index == 100:
            self.dir_index += 1
            self.file_index = 0
        dir_name = self.dir_name()
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = os.path.join(dir_name, self.file_name())
        if self.compress:
            return bz2.BZ2File(file_name + '.bz2', 'w')
        else:
            return open(file_name, 'w')

    def dir_name(self):
        char1 = self.dir_index % 26
        char2 = self.dir_index / 26 % 26
        return os.path.join(self.path_name, '%c%c' % (ord('A') + char2, ord('A') + char1))

    def file_name(self):
        return 'wiki_%02d' % self.file_index


# READER ###################################################################

tag_re = re.compile(r'(.*?)<(/?\w+)[^>]*>(?:([^<]*)(<.*?>)?)?')


def process_data(input_, output):
    global prefix

    page = []
    id_ = None
    in_text = False
    redirect = False
    for line in input_:
        line = line.decode('utf-8')
        tag_ = ''
        if '<' in line:
            m = tag_re.search(line)
            if m:
                tag_ = m.group(2)
        if tag_ == 'page':
            page = []
            redirect = False
        elif tag_ == 'id' and not id_:
            id_ = m.group(3)
        elif tag_ == 'title':
            title = m.group(3)
        elif tag_ == 'redirect':
            redirect = True
        elif tag_ == 'text':
            in_text = True
            line = line[m.start(3):m.end(3)] + '\n'
            page.append(line)
            if m.lastindex == 4:  # open-close
                in_text = False
        elif tag_ == '/text':
            if m.group(1):
                page.append(m.group(1) + '\n')
            in_text = False
        elif in_text:
            page.append(line)
        elif tag_ == '/page':
            colon = title.find(':')
            if (colon < 0 or title[:colon] in accepted_namespaces) and \
                    not redirect:
                print(id_, title.encode('utf-8'))
                sys.stdout.flush()
                wiki_document(output, id_, title, ''.join(page))
            id_ = None
            page = []
        elif tag_ == 'base':
            # discover prefix from the xml dump file
            # /mediawiki/siteinfo/base
            base = m.group(3)
            prefix = base[:base.rfind("/")]


# CL INTERFACE ############################################################

def show_help():
    print(__doc__, end=' ', file=sys.stdout)


def show_usage(script_name):
    print('Usage: %s [options]' % script_name, file=sys.stderr)


##
# Minimum size of output files
MIN_FILE_SIZE = 200 * 1024


def main():
    global keep_links, keep_sections, prefix, accepted_namespaces
    script_name = os.path.basename(sys.argv[0])

    try:
        long_opts = ['help', 'compress', 'bytes=', 'basename=', 'links', 'ns=', 'sections', 'output=', 'version']
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'cb:hln:o:B:sv', long_opts)
    except getopt.GetoptError:
        show_usage(script_name)
        sys.exit(1)

    compress = False
    file_size = 500 * 1024
    output_dir = '.'

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            show_help()
            sys.exit()
        elif opt in ('-c', '--compress'):
            compress = True
        elif opt in ('-l', '--links'):
            keep_links = True
        elif opt in ('-s', '--sections'):
            keep_sections = True
        elif opt in ('-B', '--base'):
            prefix = arg
        elif opt in ('-b', '--bytes'):
            try:
                if arg[-1] in 'kK':
                    file_size = int(arg[:-1]) * 1024
                elif arg[-1] in 'mM':
                    file_size = int(arg[:-1]) * 1024 * 1024
                else:
                    file_size = int(arg)
                if file_size < MIN_FILE_SIZE:
                    raise ValueError()
            except ValueError:
                print('%s: %s: Insufficient or invalid size' % (script_name, arg), file=sys.stderr)
                sys.exit(2)
        elif opt in ('-n', '--ns'):
            accepted_namespaces = set(arg.split(','))
        elif opt in ('-o', '--output'):
            output_dir = arg
        elif opt in ('-v', '--version'):
            print('WikiExtractor.py version:', version)
            sys.exit(0)

    if len(args) > 0:
        show_usage(script_name)
        sys.exit(4)

    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            print('Could not create: ', output_dir, file=sys.stderr)
            return

    if not keep_links:
        ignore_tag('a')

    output_splitter = OutputSplitter(compress, file_size, output_dir)
    process_data(sys.stdin, output_splitter)
    output_splitter.close()


if __name__ == '__main__':
    main()
