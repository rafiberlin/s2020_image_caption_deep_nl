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
This tool does not always detect all dependencies correctly and you might need to install some packages
manually (for example, packages like cython which are not used in the project but is used to correctly install
some dependencies). If you run into an error check the warnings and install the missing packages accordingly.

Please note that you will need to install the pytorch framework that matches your Nvidia GPU Drivers manually.

For our project, we installed pytorch and torchvision with:

```
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Please consult [pytorch.org](https://pytorch.org/get-started/locally/) to get pip instructions for other versions.

## Optional: Custom installation

If the dataset is locally available, you can create a copy of the hyperparameters file and change
the value of `root` to the desired path of the dataset. This will allow the program to search in the specified root folder for the train set, validation set and test set. Annotations should be provided in `/{path to root}/annotations`

```json
"root": "/media/user/mscoco",
```

However, this **requires** the dataset to be pre-processed first via
```
python3 main.py --prep
```

If a different version of the dataset is supposed to be used, adjust `train`, `val` and `test` to the correct version.
```json
"train": "train2017",
"val": "val2017",
"test": "test2017",
```

Be careful though as older version of the dataset might have missing pictures, which were removed later but without 
a proper update of the corresponding annotation files.

## Training and testing

If the dataset is downloaded and the glove vectors have been converted, then the training can be started via

```
python3 src/main.py --train
```

Model checkpoints will be saved in the model_storage directory,
which will be created if non-existent.

If no command-line parameter is specified, the program will only perform the model evaluation.

## Optional: Use the trained model in the Jupyter Notebook under  notebooks/run_model_evaluation.ipynb

Please download the saved model from [google drive](https://drive.google.com/file/d/1DcrmxJQWt-hVMdMJ_s1Pc030TkwBDRk2/view)

And unzip it under: `/model_storage/lp10_img256_cs224_resnet50_gru_l2_hdim512_emb300_lr0.001_wd0_epo24_bat256_do0.03_cut2_can5_with_norm_ie_s_tf.pt`

Please do not change the name of the model!

The Jupyter notebook is preconfigured to run the model to download. The results of the model evaluation and some 
predictions have been processed and saved in order to quickly get an impression of the trained model.

## Some parameters in hparams.json explained

* rnn_layers : Enables the number of layers to be stacked on the RNN
* drop_out_prob: only available when stacking RNN layers, ignore some hidden neurons with the given probabiliy. 
Set 0 to not use it.
* rnn_model : "gru" or "lstm"  are the possible values
* hidden_dim : control the dimension of the RNN hidden state. Set to 512 as in the original paper
* cnn_model : controls which pre-trained model will be used to process the images. Options are "vgg16", "mobilenet" 
and  "resnet50"
* shuffle: true or false, shuffle batches between epochs during training
* embedding_dim: the size of the word embedding to be used
* improve_embedding: if set to true, the model will try to improve the embeddings during training
* improve_cnn: if set to true, the model will try to improve the pre-trained CNN during training
* num_epochs: The number epochs used during training
* lr: learning rate. Applicable for both Adam and SGD with Nesterov Optimizer
* sgd_momentum: if set to null, the model will train using Adam. If there is a numeric value (between 0 and 1), SGD 
with Nesterov will be used
* break_training_loop_percentage: If set to 100, it will train and evaluate using the files using cleaned_captions as 
a prefix. If set to a value between 1 and 99, it will try to use a file with a prefix  {percentage}_cleaned_captions
to train / evaluate. These files contain the prefixed percentage of the original annotation files.
* annotation_without_punctuation: True or false, allows to switch with annotation files conatining punctuation or not 
(all words are lower case anyway)
* sampling_method: Enter "beam_search" to generate caption with beam search, beam_search_early_stop stops at the first finished sequence
,sample_search to sample a word based on the probability of the softmax output or null to take the word with highest probability at each_step
* beam_width: size of the beam search
* training_report_frequency: prints the total loss during training every X epochs.
* save_pending_model:  used with training_report_frequency, allow to save the corresponding model, prefixed 
with epoch number
* last_saved_model: enter the file to load a previous model. It might be useful to continue a training which 
stopped too early. However, the size of the model being loaded has to match the size of implied by the configuration. 
For example, it will not be able to load a model using embedding of size 200 before starting a training with a model using 
larger or smaller embedding dimensions.
* use_pixel_normalization: if set to true, use the values recommended in the pytorch documentation for pretrained model.
* image_size: value between 256 and 640. This is the size of the image being used. Regardless of the defined size, 
a random center crop of size 224 by 224 is done during training (and a regular center crop for evaluation and test)
* crop_size: used for randomly cropping the image by size crop_size*crop_size. Minimum value is 224.
* caption_number; the number of captions to be used by the model. Values between 1 and 5.
* cutoff: Indicates the minimum number of occurrences for a caption word to avoid being flagged as unknown.
* clip_grad: Allows to clip the gradient, if the total loss (or rather the calculated model weights) becomes 
an invalid number. Recommended value is 1.0. To skip its use, set the parameter to null
* weight_decay: used for both optimizer, allows to use regularization. Set it to 0 to disable it.
* compute_val_loss: parameter dependent of training_report_frequency. It will monitor the total loss on the validation 
set at the frequency defined in training_report_frequency. 
* gold_eval_with_original: If set to true, the original reference labels, which does not contain any <UNK> token to 
flag unknown words will be used to calculate the Bleu Score.
* save_eval_results: If set to true, it will allow to store evaluation results under model storage
* keep_best_val_loss:   Used with "compute_val_loss". If the total validation loss starts to grow bigger, the 
training will stop and the last model before this loss increase will be saved.
All these parameters will influence the name of the saved model.
* keep_best_total_loss: dependent on training_report_frequency. If set to true, training loop will be stopped early to 
reload the previous model, if the total loss stops decreasing
* teacher_forcing: enables teacher forcing. Do not set to false as the training without teacher forcing suffers from 
memory issues.

For example, based on the training configuration, a valid model name could be: 

`lp1_img256_mobilenet_gru_l4_hdim512_emb300_lr0.001_wd1e-05_epo300_bat32_do0.35_cut2_can3_with_norm_ie_s_ic.pt`

