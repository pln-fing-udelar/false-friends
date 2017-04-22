FROM python:3.5

RUN pip install Cython

RUN apt-get update
RUN apt-get install -y \
    apertium \
    apertium-es-pt \
    libicu-dev

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader wordnet omw

COPY . /usr/src/app

VOLUME ["/usr/src/app/resources/big"]
