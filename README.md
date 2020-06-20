## Installing

* Download the [COCO dataset] and place it into the data directory.
* Download [GloVe Vectors](http://nlp.stanford.edu/data/glove.6B.zip)

```
wget http://nlp.stanford.edu/data/glove.6B.zip -O data/glove.6B.zip
unzip data/glove.6B.zip -d data
```

Convert the GloVe embeddings into binary word2vec format to allow faster loading
```
python3 utils/glove_conv.py
```

Install python libraries

```
pip3 install -r requirements.txt
```

Requirements have been automatically generated using [pigar](https://github.com/damnever/pigar).

## Running

```
python3 src/main.py
```