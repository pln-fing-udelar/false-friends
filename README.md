# False Friends

This project is about distinguishing true and false friends between Spanish and Portuguese. To run the code, just
execute the following (and follow the instructions from there):

```shell
./falsefriends.py --help
```

## Similar words in Wikipedia's

Similar words in Wikipedia's can be studied also running:

```shell
ipython -i scripts/comparesimilar.py
```

With the `-i` flag, a interactive iPython shell is ready to be used after the script execution.

## Run with Docker

```shell
docker build -t false-friends .
docker run -ti -v $PWD/resources/big:/usr/src/app/resources/big false-friends /bin/bash
```

