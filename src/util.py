import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dset
import gensim
import numpy as np
import preprocessing as prep
from pycocotools.coco import COCO
from PIL import Image

class CocoDatasetWrapper(Dataset):

    def __init__(self, cocodaset, vectorizer, caption_number=5):
        self.cocodaset = cocodaset
        self.vectorizer = vectorizer
        if caption_number > 5:
            caption_number = 5
        self.caption_number = caption_number

    @classmethod
    def _get_transform_pipeline_and_shuffle(cls, hparams, dataset_name):
        img_size = hparams['image_size']
        # Most on the example in pytorch have this minimum size before cropping
        assert img_size >= 256
        cropsize = hparams["crop_size"]
        # Most on the example in pytorch have this minimum crop size for random cropping
        assert cropsize >= 224

        if dataset_name == "train":
            shuffle = hparams["shuffle"]
            if hparams["use_pixel_normalization"]:
                transform_pipeline = transforms.Compose([prep.CenteringPad(),
                                                         transforms.Resize(
                                                             (img_size, img_size)),
                                                         transforms.RandomCrop(
                                                             cropsize),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                                              # recommended resnet config
                                                                              (0.229, 0.224, 0.225))
                                                         ])
            else:
                transform_pipeline = transforms.Compose([prep.CenteringPad(),
                                                         transforms.Resize(
                                                             (img_size, img_size)),
                                                         transforms.RandomCrop(
                                                             cropsize),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor()])
        else:
            shuffle = False
            if hparams["use_pixel_normalization"]:
                transform_pipeline = transforms.Compose([prep.CenteringPad(),
                                                         transforms.Resize(
                                                             (img_size, img_size)),
                                                         transforms.CenterCrop(
                                                             cropsize),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                                              # recommended resnet config
                                                                              (0.229, 0.224, 0.225))
                                                         ])
            else:
                transforms.Compose([prep.CenteringPad(),
                                    transforms.Resize((img_size, img_size)),
                                    transforms.CenterCrop(cropsize),
                                    transforms.ToTensor()
                                    ])
        return transform_pipeline, shuffle

    @classmethod
    def create_dataloader(cls, hparams, c_vectorizer, dataset_name="train2017", image_dir=None):
        """
        For dataset_name="train", the data will be shuffled, randomly flipped and cropped. For the rest, just a center
        crop without shuffling
        :param hparams:
        :param c_vectorizer:
        :param dataset_name:
        :param image_dir:
        :return:
        """
        train_file = hparams[dataset_name]
        if image_dir == None:
            image_dir = os.path.join(hparams['root'], train_file)
        else:
            image_dir = os.path.join(hparams['root'], image_dir)
        caption_file_path = prep.get_correct_annotation_file(
            hparams, dataset_name)
        print("Image dir:", image_dir)
        print("Caption file path:", caption_file_path)

        # rgb_stats = prep.read_json_config(hparams["rgb_stats"])
        # stats_rounding = hparams["rounding"]
        # rgb_stats = {"mean": [0.31686973571777344, 0.30091845989227295, 0.27439242601394653],
        #              "sd": [0.317791610956192, 0.307492196559906, 0.3042858839035034]}
        # rgb_mean = tuple([round(m, stats_rounding) for m in rgb_stats["mean"]])
        # rgb_sd = tuple([round(s, stats_rounding) for s in rgb_stats["mean"]])
        transform_pipeline, shuffle = cls._get_transform_pipeline_and_shuffle(
            hparams, dataset_name)

        coco_train_set = dset.CocoDetection(root=image_dir,
                                            annFile=caption_file_path,
                                            transform=transform_pipeline
                                            )

        caption_number = hparams["caption_number"]
        coco_dataset_wrapper = CocoDatasetWrapper(
            coco_train_set, c_vectorizer, caption_number)
        batch_size = hparams["batch_size"]

        train_loader = torch.utils.data.DataLoader(coco_dataset_wrapper, batch_size=batch_size, pin_memory=True,
                                                   shuffle=shuffle)
        return train_loader

    @classmethod
    def transform_batch_for_training(cls, batch, device="cpu"):
        """

        :param batch:
        :return: a tuple of 3 element: the images, the in-vectorized captions and
                the out-vectorized captions for the loss function
        """
        return batch[0].to(device), batch[2][0].to(device), batch[2][1].to(device)

    def __len__(self):
        return self.cocodaset.__len__()

    def __getitem__(self, index):
        image, captions = self.cocodaset.__getitem__(index)
        # it seams like we always get 5 different captions for an image...
        num_captions = len(captions)
        # self.vectorizer.max_sequence_length - 1 because in label and out labels are shifted by 1 to match
        # for example, if the last real word in a caption is cat, the expected output caption is <end>...
        vectorized_captions_in = torch.zeros(
            (num_captions, self.vectorizer.max_sequence_length - 1), dtype=torch.long)
        vectorized_captions_out = torch.zeros(
            (num_captions, self.vectorizer.max_sequence_length - 1), dtype=torch.long)
        for i, caption_reviewer in enumerate(captions):
            c = self.vectorizer.vectorize(captions[i]["caption"])
            vectorized_captions_in[i], vectorized_captions_out[i] = tuple(
                map(torch.from_numpy, c))

        # only use 5 or less captions to be able to use faster vectorized operations
        # avoid exceptions in the collate function in the fetch part of the dataloader
        return image, captions[:self.caption_number], (
            vectorized_captions_in[:self.caption_number], vectorized_captions_out[:self.caption_number])


class CocoDatasetAnnotation(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vectorizer, hparams, annotation_ids=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        if annotation_ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = annotation_ids
        self.vectorizer = vectorizer
        transform, _ = CocoDatasetWrapper._get_transform_pipeline_and_shuffle(hparams, "train")
        self.transform = transform
    @classmethod
    def _get_transform_pipeline_and_shuffle(cls, hparams, dataset_name):
        img_size = hparams['image_size']
        # Most on the example in pytorch have this minimum size before cropping
        assert img_size >= 256
        cropsize = hparams["crop_size"]
        # Most on the example in pytorch have this minimum crop size for random cropping
        assert cropsize >= 224
        transform_pipeline = None
        if dataset_name == "train":
            shuffle = hparams["shuffle"]
            if hparams["use_pixel_normalization"]:
                transform_pipeline = transforms.Compose([prep.CenteringPad(),
                                                         transforms.Resize(
                                                             (img_size, img_size)),
                                                         transforms.RandomCrop(
                                                             cropsize),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                                              # recommended resnet config
                                                                              (0.229, 0.224, 0.225))
                                                         ])
            else:
                transform_pipeline = transforms.Compose([prep.CenteringPad(),
                                                         transforms.Resize(
                                                             (img_size, img_size)),
                                                         transforms.RandomCrop(
                                                             cropsize),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor()])
        else:
            shuffle = False
            if hparams["use_pixel_normalization"]:
                transform_pipeline = transforms.Compose([prep.CenteringPad(),
                                                         transforms.Resize(
                                                             (img_size, img_size)),
                                                         transforms.CenterCrop(
                                                             cropsize),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                                              # recommended resnet config
                                                                              (0.229, 0.224, 0.225))
                                                         ])
            else:
                transforms.Compose([prep.CenteringPad(),
                                    transforms.Resize((img_size, img_size)),
                                    transforms.CenterCrop(cropsize),
                                    transforms.ToTensor()
                                    ])
        return transform_pipeline, shuffle

    @classmethod
    def create_dataloader(cls, hparams, c_vectorizer, dataset_name="train2017", image_dir=None, annotation_ids=None):
        """
        For dataset_name="train", the data will be shuffled, randomly flipped and cropped. For the rest, just a center
        crop without shuffling
        :param hparams:
        :param c_vectorizer:
        :param dataset_name:
        :param image_dir:
        :return:
        """
        train_file = hparams[dataset_name]
        if image_dir == None:
            image_dir = os.path.join(hparams['root'], train_file)
        else:
            image_dir = os.path.join(hparams['root'], image_dir)
        caption_file_path = prep.get_correct_annotation_file(
            hparams, dataset_name)
        print("Image dir:", image_dir)
        print("Caption file path:", caption_file_path)

        transform_pipeline, shuffle = cls._get_transform_pipeline_and_shuffle(
            hparams, dataset_name)

        coco_annotation_loader = CocoDatasetAnnotation(image_dir, caption_file_path, c_vectorizer, hparams, annotation_ids=annotation_ids)
        batch_size = hparams["batch_size"]

        train_loader = torch.utils.data.DataLoader(coco_annotation_loader, batch_size=batch_size, pin_memory=True,
                                                   shuffle=shuffle)
        return train_loader

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption (string) to word ids.
        in_caption, out_caption = self.vectorizer.vectorize(caption)
        return image, in_caption, out_caption

    def __len__(self):
        return len(self.ids)

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def create_embedding(hparams, c_vectorizer, padding_idx=0):
    vocabulary_size = len(c_vectorizer.get_vocab())
    if hparams["use_glove"]:
        print("Loading glove vectors...")
        glove_path = os.path.join(hparams['root'], hparams['glove_embedding'])
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(
            glove_path, binary=True)
        glove_embedding = np.zeros((vocabulary_size, glove_model.vector_size))
        token2idx = {
            word: glove_model.vocab[word].index for word in glove_model.vocab.keys()}
        for word in c_vectorizer.get_vocab()._token_to_idx.keys():
            i = c_vectorizer.get_vocab().lookup_token(word)
            if i != padding_idx:
                if word in token2idx:
                    glove_embedding[i, :] = glove_model.vectors[token2idx[word]]
                else:
                    # From NLP with pytorch, it should be better to init the unknown tokens
                    embedding_i = torch.ones(1, glove_model.vector_size)
                    torch.nn.init.xavier_uniform_(embedding_i)
                    glove_embedding[i, :] = embedding_i
            else:
                embedding_i = torch.zeros(1, glove_model.vector_size)
                glove_embedding[i, :] = embedding_i
        embed_size = glove_model.vector_size
        if hparams["embedding_dim"] < embed_size:
            embed_size = hparams["embedding_dim"]

        embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(glove_embedding[:, :embed_size]))
        # embedding = nn.Embedding.from_pretrained(
        #     torch.FloatTensor(glove_embedding[:, :embed_size]))
        embedding.weight.requires_grad = hparams["improve_embedding"]
        print("GloVe embedding size:", glove_model.vector_size)
    else:
        embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                 embedding_dim=hparams["embedding_dim"], padding_idx=padding_idx)
    return embedding


def create_model_name(hparams):
    """
    Creates a model name that encodes the training parameters
    :param hparams:
    :return:
    """

    root_name, extension = hparams["model_name"].split(".")
    norm = ""
    if hparams['use_pixel_normalization']:
        norm = "_with_norm"
    clip_grad = ""
    if hparams['clip_grad']:
        clip_grad = f"_cg{hparams['clip_grad']}"
    improve_embeddings = ""
    if hparams['improve_embedding']:
        improve_embeddings = "_ie"
    shuffle = ""
    if hparams['shuffle']:
        shuffle = "_s"
    improve_cnn = ""
    if hparams['improve_cnn']:
        improve_cnn = "_ic"
    sgd_momentum = ""
    if hparams['sgd_momentum']:
        sgd_momentum = f"_sgdm{hparams['sgd_momentum']}"
    without_punct = ""
    if hparams["annotation_without_punctuation"]:
        without_punct = "_wp"
    teacher_forcing = ""
    if hparams["teacher_forcing"]:
        without_punct = "_tf"
    padding_idx=""
    if hparams["use_padding_idx"]:
        padding_idx="_pad"
    model_name = f"lp{hparams['break_training_loop_percentage']}_img{hparams['image_size']}_cs{hparams['crop_size']}_{hparams['cnn_model']}_{hparams['rnn_model']}_l{hparams['rnn_layers']}{root_name}hdim{str(hparams['hidden_dim'])}_emb{str(hparams['embedding_dim'])}_lr{str(hparams['lr'])}_wd{str(hparams['weight_decay'])}{sgd_momentum}_epo{str(hparams['num_epochs'])}_bat{str(hparams['batch_size'])}_do{str(hparams['drop_out_prob'])}_cut{str(hparams['cutoff'])}_can{str(hparams['caption_number'])}{norm}{clip_grad}{improve_embeddings}{shuffle}{improve_cnn}{without_punct}{teacher_forcing}{padding_idx}.{extension}"
    return model_name
