# False Friends

This project is about distinguishing true and false friends between Spanish and Portuguese. To run the code, just
execute the following (and follow the instructions from there):

```shell
./falsefriends.py --help
```

## Docker

To open a bash shell in the context of the project, use [this docker image](https://hub.docker.com/plnfingudelar/false-friends):

```shell
docker run -ti pln-fing-udelar/false-friends /bin/bash
```

To map a local directory with resources with the container, for example run:

```shell
docker run -ti -v $PWD/resources/big:/usr/src/app/resources/big pln-fing-udelar/false-friends /bin/bash
```

If you want to build the image:

```shell
docker build -t false-friends .
```

## Non-Docker way

To install dependencies:

```shell
pip install Cython # Needed to **install** "word2vec" package. 
pip install -r requirements.txt
```

## Similar words in Wikipedia's

Similar words in Wikipedia's can be studied also running:

```shell
ipython -i scripts/comparesimilar.py
```

With the `-i` flag, a interactive iPython shell is ready to be used after the script execution.

## Citation

If you use this work in your research, please cite us:

```
@InProceedings{W18-3903,
  author = 	"Castro, Santiago
		and Bonanata, Jairo
		and Ros{\'a}, Aiala",
  title = 	"A High Coverage Method for Automatic False Friends Detection for Spanish and Portuguese",
  booktitle = 	"Proceedings of the Fifth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2018)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"29--36",
  location = 	"Santa Fe, New Mexico, USA",
  url = 	"http://aclweb.org/anthology/W18-3903"
}
```
