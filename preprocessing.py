from collections import OrderedDict, Counter
import torchvision.datasets as dset
import json
import nltk
from pathlib import Path
import re
from torchvision.transforms.functional import pad
from torchvision import transforms
import numbers
import os
from nltk.tokenize import word_tokenize

def create_json_config(params, file_path, indent=3):
    with open(file_path, 'w') as json_file:
        json.dump(params, json_file)


def read_json_config(file_path):
    '''

    :param file_path:
    :return:
    '''
    #data = json.load(open(file_path), object_pairs_hook=OrderedDict)
    data = json.load(open(file_path))
    return data



class ImageSizeStats(object):
    """
    Class that gives some statistics about image sizes
    """

    def __init__(self, coco_dataset, create_json= False):

        self.dataset = coco_dataset
        all_shapes = [(self.dataset.coco.imgs[i]["height"], self.dataset.coco.imgs[i]["width"]) for i in self.dataset.coco.imgs]
        if create_json:
            create_json_config(all_shapes, "shapes.json")

        self.c = Counter(all_shapes)
        self.avg_img_size = self.__calculate_avg_image_size()

    def get_avg_img_size(self):
        return self.avg_img_size
    def most_common(self, n=1):
        return self.c.most_common(n)
    def least_common(self, n=1):
        return self.c.most_common()[-n]


    def __calculate_avg_image_size(self):
        set_size = len(self.dataset)
        h_m = 0.0
        w_m = 0.0
        for k in self.c.keys():
            h, w = k
            h_m += h * self.c[k] / set_size
            w_m += w * self.c[k] / set_size

        return h_m, w_m

def print_img_infos_datasets():
    DATASET_FILE_PATHS_CONFIG = "dataset_file_args.json"
    HYPER_PARAMETER_CONFIG = "hyper_parameters.json"

    file_args = read_json_config(DATASET_FILE_PATHS_CONFIG)
    hyper_parameters = read_json_config(HYPER_PARAMETER_CONFIG)

    # TODO create a testing split, there is only training and val currently...
    coco_train_set = dset.CocoDetection(root=file_args["train"]["img"],
                                        annFile=file_args["train"]["capt"],
                                        transform=transforms.Compose([transforms.ToTensor()])
                                        )

    coco_val_set = dset.CocoDetection(root=file_args["val"]["img"],
                                        annFile=file_args["val"]["capt"],
                                        transform=transforms.Compose([transforms.ToTensor()])
                                        )

    coco_test_set = dset.CocoDetection(root=file_args["test"]["img"],
                                        annFile=file_args["test"]["capt"],
                                        transform=transforms.Compose([transforms.ToTensor()])
                                        )
    print("train")
    print_img_infos(coco_train_set)
    print("test")
    print_img_infos(coco_test_set)
    print("val")
    print_img_infos(coco_val_set)

def print_img_infos(dataset):
    stats = ImageSizeStats(dataset)
    print("Most Common resolution", stats.most_common())
    print("Least Common resolution", stats.least_common())
    print("Average resolution", stats.get_avg_img_size())


def preprocess_text(text):
    """
    Removes punctuation and add lower case
    :param text:
    :return:
    """
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalpha()]
    text = " ".join(text)
    return text






class CenteringPad(object):
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
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        self.padding = self.get_padding(img)
        return pad(img, self.padding, self.fill, self.padding_mode)

    def get_padding(self, image):
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
        return self.__class__.__name__ + f'(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)

def create_list_of_captions(caption_dir, file_name, save_file_path=None):
    """
    Given a caption json file for the COCO dataset, lower case the labels
    and add space before and after punctuation, Preprocessing function from
    "Natural Language with Pytorch", chapter 3.
    :param file_path:
    :param save_file_path:
    :return:
    """

    if save_file_path is None:
        #transforms captions_train2017.json to captions_train2017_label_only.json
        save_file_path = "_label_only.".join(file_name.split("."))

    file_path = os.path.join(caption_dir, file_name)
    cleaned_file_path = os.path.join(caption_dir, "cleaned_"+file_name)
    captions = read_json_config(file_path)

    caption_file = Path(save_file_path)
    if not caption_file.is_file():
        cleaned_captions = []
        for idx, caption in enumerate(captions["annotations"]):
            cleaned_caption = preprocess_text(caption["caption"])
            cleaned_captions.append(cleaned_caption)
            captions[idx] = cleaned_caption
        create_json_config(cleaned_captions, save_file_path, 0)
        create_json_config(captions, cleaned_file_path, 0)
    else:
        cleaned_captions = read_json_config(save_file_path)
    return cleaned_captions

def clean_caption_annotations(annotation_dir, annotation_list):
    for annotation in annotation_list:
        create_list_of_captions(annotation_dir, annotation)


if __name__ == '__main__':
    clean_caption_annotations("./data/annotations/", ["captions_train2017.json", "captions_val2017.json"])





