import torchvision.datasets as dset
from pathlib import Path
from torchvision.transforms.functional import pad
from torchvision import transforms
import torch
import numbers
import os
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
import numpy as np
import funcy
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
from zipfile import ZipFile
from argparse import Namespace
import torch.utils.data
from pycocotools.coco import COCO

if not torch.cuda.is_available():
    DEVICE = "cpu"
else:
    DEVICE = "cuda:0"


def create_json_config(params, file_path):
    """
    :param params: the array object to be transformed to a JSON
    :param file_path: full path name of the JSON file
    :return:
    """
    with open(file_path, 'w') as json_file:
        json.dump(params, json_file)


def read_json_config(file_path):
    """
    Returns the JSON file as Python object
    :param file_path: full path name of the JSON file
    :return: Returns the JSON file as Python object
    """

    with open(file_path, 'r') as f:
        return json.load(f)


def preprocess_text(text, remove_punctuation=True):
    """
    Removes punctuation and add lower case

    :param text: Text to preprocess
    :param remove_punctuation: Only return alphabetical words
    :return: 
    """
    text = text.lower()
    text = word_tokenize(text)
    if remove_punctuation:
        text = [word for word in text if word.isalpha()]
    text = " ".join(text)
    return text


class CenteringPad:
    """
    Pad class to deal with varying sizes. Strategy for all images which does not have the max resolution of the
    data set 640 by 640, we center the image and pad. Target resolution needs to be square.
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850
    """

    def __init__(self, fill=0, padding_mode='constant', target_resolution=(640, 640)):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        assert target_resolution[0] == target_resolution[1]
        self.padding = None
        self.fill = fill
        self.padding_mode = padding_mode
        self.target_resolution = target_resolution

    def __call__(self, img):
        """
        Returns the padded image

        :param img: Image to be padded
        :return: Padded image
        """
        self.padding = self.get_padding(img)
        return pad(img, self.padding, self.fill, self.padding_mode)

    def get_padding(self, image):
        """
        Given the image, automatically calculates the padding needed on each border.

        :param image: image to pad
        :return: Padded image
        """
        if image.size == self.target_resolution:
            return 0
        w, h = image.size
        # We want 640 by 640 centered images in any case
        max_wh = self.target_resolution[0]
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

        self.padding = padding
        return padding

    def __repr__(self):
        return self.__class__.__name__ + f"(padding={0}, fill={1}, padding_mode={2})".format(self.padding, self.fill,
                                                                                             self.padding_mode)


def get_current_images_id(hparams, dataset_name):
    """
    Get the list of image ids corresponding to break_training_loop_percentage in hparams.
    It helps loading less captions, embeddings in create_list_of_captions_and_clean() , hence getting more samples per batches.

    :param hparams: Hyperparameters
    :param dataset_name: Dataset name to get images from
    :return: List of image indices
    """

    train_file = hparams[dataset_name]
    image_dir = os.path.join(hparams['root'], train_file)
    caption_file_path = get_cleaned_captions_path(hparams, train_file)
    img_size = hparams["image_size"]
    transform_pipeline = transforms.Compose([CenteringPad(),
                                             transforms.Resize(
                                                 (img_size, img_size)),
                                             transforms.ToTensor()])

    coco_train_set = dset.CocoDetection(root=image_dir,
                                        annFile=caption_file_path,
                                        transform=transform_pipeline
                                        )
    train_loader = torch.utils.data.DataLoader(
        coco_train_set, batch_size=hparams["batch_size"])
    break_training_loop_percentage = hparams["break_training_loop_percentage"]
    break_training_loop_idx = max(
        int(len(train_loader) * break_training_loop_percentage / 100) - 1, 0)

    img_list = []
    for idx, sample in enumerate(tqdm(train_loader)):
        img_list.extend(sample[1][0]["image_id"].tolist())
        if idx == break_training_loop_idx:
            break
    return img_list


def get_correct_annotation_file(hparams, name, remove_punctuation=True):
    """
    Contrived logic, based on the percentage for breaking the loop,
    pick the correct file name...
    If break_training_loop_percentage = 100, it will pick cleaned_captions_train2017.json
    If break_training_loop_percentage = 10, it will try to pick 10_cleaned_captions_train2017.json

    :param hparams: Hyperparameters
    :param name: Train/Test/Val
    :param remove_punctuation: Only keep alphabetical words
    :return: Path to the annotation file
    """
    percentage = hparams["break_training_loop_percentage"]
    caption_dir = os.path.join(hparams['root'], "annotations")
    punct = ""
    if not remove_punctuation:
        punct = "punct_"
    use_reduced_set = False
    if percentage < 100:
        save_file_path = os.path.join(
            caption_dir, f"{percentage}_cleaned_captions_{punct + hparams[name]}.json")
        caption_file = Path(save_file_path)
        use_reduced_set = caption_file.is_file()
    if not use_reduced_set:
        save_file_path = os.path.join(
            caption_dir, f"cleaned_captions_{punct + hparams[name]}.json")
        caption_file = Path(save_file_path)
        file_available = caption_file.is_file()
    else:
        file_available = True
    if file_available:
        return save_file_path
    print("Warning, no annotation found!")
    return None


def get_captions(hparams, name):
    """
    Only extracts the needed captions set in caption_number. Might reduce memory footprint for embeddings

    :param hparams: Hyperparameters
    :param name: Train/Test/Val
    :return: List of captions for the given dataset name, and a list of annotation ids
            needed to initialize the util.CocoDatasetAnnotation dataloader correctly.
    """

    captions_number = hparams["caption_number"]
    caption_file_path = get_correct_annotation_file(hparams, name)

    coco_caps = COCO(caption_file_path)
    img_ids = coco_caps.getImgIds()

    result = []
    annotation_ids = []
    for img in img_ids:
        # only load captions used during training
        ann_ids = coco_caps.getAnnIds(img)[:captions_number]
        anns = coco_caps.loadAnns(ann_ids)
        annotation_ids.extend(ann_ids)
        result.extend([a["caption"] for a in anns])
    return result, annotation_ids


def create_list_of_captions_and_clean(hparams, name, img_list=None, remove_punctuation=True):
    """
    Given a caption json file for the COCO dataset, lower case the labels
    and add space before and after punctuation, Preprocessing function from
    "Natural Language with Pytorch", chapter 3.

    :param hparams: Hyperparameters
    :param name: Name of the dataset (Train/Test/Val)
    :param img_list: Filter captions for certain images
    :param remove_punctuation: Removes punctuation
    :return: List of cleaned captions
    """

    punct = ""
    if not remove_punctuation:
        punct = "punct_"

    file_path = get_cleaned_captions_path(hparams, punct + hparams[name])

    if not Path(file_path).is_file():
        file_path = get_captions_path(hparams, hparams[name])
    save_file_path = get_correct_annotation_file(
        hparams, name, remove_punctuation)
    # Fallback on original files

    if not Path(file_path).is_file() and not save_file_path:
        raise Exception("Neither cleaned version of annotation files under nor original avalaible under : ",
                        hparams["root"])
    # If the cleaned version does not exist, create it
    if not save_file_path:
        save_file_path = os.path.join(os.path.join(hparams['root'], "annotations"),
                                      f"cleaned_captions_{punct + hparams[name]}.json")
        with open(file_path, "r") as f:
            captions = json.load(f)
            print("Cleaning captions...")
            cleaned_captions = []
            for idx, caption in enumerate(tqdm(captions["annotations"])):
                if img_list is None or caption["image_id"] in img_list:
                    cleaned_caption = preprocess_text(
                        caption["caption"], remove_punctuation)
                    cleaned_captions.append(cleaned_caption)
                    captions["annotations"][idx]["caption"] = cleaned_caption

            with open(save_file_path, "w") as f:
                json.dump(captions, f)

            return cleaned_captions
    else:
        print("Reading pre-cleaned captions...")
        with open(save_file_path, "r") as f:
            captions = json.load(f)
            cleaned_captions = []
            for caption in tqdm(captions["annotations"]):
                if img_list is None or caption["image_id"] in img_list:
                    cleaned_captions.append(caption["caption"])
            return cleaned_captions


def clean_caption_annotations(hparams, annotation_list, remove_punctuation=True):
    """
    Creates the cleaned caption annotations

    :param hparams: Hyperparameters
    :param annotation_list: List of annotation names
    :param remove_punctuation: Removes punctuation
    """
    for annotation in annotation_list:
        create_list_of_captions_and_clean(
            hparams, annotation, None, remove_punctuation)


def get_captions_path(hparams, dataset):
    """
    Returns a full path to an annotation

    :param hparams: Project parameters
    :param dataset: dataset name (train2017, test2017, val2017)
    :return: Returns a full path to an annotation
    """
    return f"{hparams['root']}/annotations/captions_{dataset}.json"


def get_cleaned_captions_path(hparams, dataset):
    """
    Returns a full path to a cleaned annotation

    :param hparams: Project parameters
    :param dataset: dataset name (train2017, test2017, val2017)
    :return: Returns a full path to a cleaned annotation
    """

    return f"{hparams['root']}/annotations/cleaned_captions_{dataset}.json"


def save_coco(file, info, licenses, images, annotations):
    """
    Downloaded from github.com/akarazniewicz/cocosplit.git@master and modified to just handle annotations
    Function helping to create new data split from an original COCO Dataset

    :param file: File to write to
    :param info: Info
    :param licenses: Licenses
    :param images: Images
    :param annotations: Annotations
    """

    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations}, coco, sort_keys=True)


def filter_annotations(annotations, images):
    """
    Downloaded from github.com/akarazniewicz/cocosplit.git@master and modified to just handle annotations
    Function helping to create new data split from an original COCO Dataset

    :param annotations:
    :param images:
    :return:
    """
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def create_cocosplit(args):
    """
    Downloaded from github.com/akarazniewicz/cocosplit.git@master and modified to just handle annotations
    Function used to create new data split from an original COCO Dataset

    :param args:
    :return:
    """
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(
            lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(
                lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=args.split, shuffle=True)
        if args.percentage < 0:
            args.percentage = 0
        if args.percentage > 100:
            args.percentage = 100
        break_x_idx = max(int(len(x) * args.percentage / 100) - 1, 0)
        break_y_idx = max(int(len(y) * args.percentage / 100) - 1, 0)
        save_coco(args.train, info, licenses, x[0:break_x_idx], filter_annotations(
            annotations, x[0:break_x_idx]))
        save_coco(args.test, info, licenses, y[0:break_y_idx], filter_annotations(
            annotations, y[0:break_y_idx]))

        print("Saved {} entries in {} and {} in {}".format(
            len(x), args.train, len(y), args.test))


def reduce_cocosplit(args):
    """
    Downloaded from github.com/akarazniewicz/cocosplit.git@master
    Function used to reduce pre-cleaned COCO annotation to keep only a certain percentage of
    COCO examples with annotations.

    :param args:
    :return:
    """
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']

        images_with_annotations = funcy.lmap(
            lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(
                lambda i: i['id'] not in images_with_annotations, images)

        if args.percentage < 0:
            args.percentage = 0.0
        if args.percentage > 100:
            args.percentage = 100.0

        x, _ = train_test_split(
            images, train_size=args.percentage / 100, shuffle=True)

        save_coco(args.train, info, licenses, x,
                  filter_annotations(annotations, x))

        print("Saved")


def set_seed_everywhere(seed):
    """
    From NLP with Pytorch book. Apply the same seed number on all framework in use to
    make results consistents over several runs

    :param seed: Global seed to set
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_unpack_zip(zipurl, storage_directory):
    """
    From https://svaderia.github.io/articles/downloading-and-unzipping-a-zipfile/
    Download a file from a URL and store it under desired diretory

    :param zipurl: The url of the file
    :param storage_directory: Directory to save zip to
    """
    print("Download:", zipurl)
    zipresp = urlopen(zipurl)
    # Create a new file on the hard drive
    temp_path = os.path.join(storage_directory, "tempfile.zip")
    tempzip = open(temp_path, "wb")
    # Write the contents of the downloaded file into the new file
    tempzip.write(zipresp.read())
    tempzip.close()
    zf = ZipFile(temp_path)
    # Extract its contents into <extraction_path>
    # note that extractall will automatically create the path
    zf.extractall(path=storage_directory)
    # close the ZipFile instance
    zf.close()
    os.remove(temp_path)
    print("Download completed:", zipurl)


def reduce_annotation_size(annotation_directory="./data/annotations", cleaned_file_prefix="cleaned_captions_",
                           final_percentage=10):
    """
    Function used to reduce pre-cleaned COCO annotation to keep only a certain percentage of
    COCO examples with annotations.

    :param annotation_directory: Directory where annotations are stored
    :param cleaned_file_prefix: prefix of the cleaned annotations
    :param final_percentage: final proportion of COCO examples to be kept
    """
    file_suffixes = ["train2017.json", "val2017.json", "test2017.json"]

    for fs in file_suffixes:
        arg_for_split = Namespace(annotations=f'{annotation_directory}/{cleaned_file_prefix + fs}',
                                  having_annotations=False,
                                  train=f'{annotation_directory}/{final_percentage}_{cleaned_file_prefix + fs}',
                                  percentage=final_percentage)
        reduce_cocosplit(arg_for_split)
        print("Annotations created:",
              f'{annotation_directory}/{final_percentage}_{cleaned_file_prefix + fs}')
