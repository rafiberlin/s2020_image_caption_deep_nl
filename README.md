## Installing

* Download the [COCO dataset images from 2017](http://cocodataset.org/#download) and [GloVe Embedding Vectors](http://nlp.stanford.edu/data/glove.6B.zip) and place them into the data directory. The needed annotations are already included in this repository. Please note that the described command will not only download the dataset but will also start the model training and evaluation.

```
python3 src/main.py --download
```
* A faster alternative to download the needed data is to use wget available on Unix systems.

```
wget http://images.cocodataset.org/zips/train2017.zip -O data/train2017.zip
unzip data/train2017.zip -d data

wget http://images.cocodataset.org/zips/val2017.zip -O data/val2017.zip
unzip data/val2017.zip -d data

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/annotations_trainval2017.zip
unzip data/annotations_trainval2017.zip -d data

wget http://nlp.stanford.edu/data/glove.6B.zip -O data/glove.6B.zip
unzip data/glove.6B.zip -d data
```

* If you decided to download the data with wget, please note that you need to run this script before starting the training and evaluation:
```
python3 utils/glove_conv.py
```

* Install python libraries

```
pip3 install -r requirements.txt
```

Requirements have been automatically generated using [pigar](https://github.com/damnever/pigar).

## Running

```
python3 src/main.py
```

Model checkpoints will be saved in the model_storage directory,
which will be created if non-existent.