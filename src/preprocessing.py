from collections import Counter
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
import model
import torch.utils.data

if not torch.cuda.is_available():
    DEVICE="cpu"
else:
    DEVICE="cuda:0"
def create_json_config(params, file_path, indent=3):
    with open(file_path, 'w') as json_file:
        json.dump(params, json_file)


def read_json_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


class ImageSizeStats(object):
    """
    Class that gives some statistics about image sizes
    """

    def __init__(self, coco_dataset, create_json=False):

        self.dataset = coco_dataset
        all_shapes = [(self.dataset.coco.imgs[i]["height"], self.dataset.coco.imgs[i]["width"]) for i in
                      self.dataset.coco.imgs]
        if create_json:
            create_json_config(all_shapes, "../shapes.json")

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

    def __calculate_avg_image_size(self):
        set_size = len(self.dataset)
        h_m = 0.0
        w_m = 0.0
        for k in self.c.keys():
            h, w = k
            h_m += h * self.c[k] / set_size
            w_m += w * self.c[k] / set_size

        return h_m, w_m

    def get_RGB_mean(self, image_size=(640, 640), batch_size=300):

        mean = torch.zeros(3, device=DEVICE)
        for i_batch, sample_batched in enumerate(torch.utils.data.DataLoader(self.dataset, batch_size)):
            imgs = sample_batched[0].to(DEVICE)
            mean += imgs.sum((2, 3)).sum(0)
        mean = mean / (len(self.dataset) * image_size[0] * image_size[1])
        create_json_config({"mean": [mean[0].item(), mean[1].item(), mean[2].item()]}, "mean.json")
        return mean

    def get_RGB_sd(self, mean, image_size=(640, 640), batch_size=300):
        sd = torch.zeros(3, device=DEVICE)
        for i_batch, sample_batched in enumerate(torch.utils.data.DataLoader(self.dataset, batch_size)):
            imgs = sample_batched[0].to(DEVICE)
            # mean.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) => adapt to same number of dimensions
            # to apply the substraction filter wise and element wise
            s = ((imgs - mean.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)).pow(2))
            # Then just apply the formula for standard deviation
            sd += s.sum((2, 3)).sum(0)
        sd = torch.sqrt(sd / ((len(self.dataset)) * image_size[0] * image_size[1]))
        return sd

    def get_RGB_mean_sd(self, image_size=(640, 640), batch_size=100):
        # mean = torch.FloatTensor([0.31686973571777344, 0.30091845989227295, 0.27439242601394653]).to("cuda:0")
        mean = self.get_RGB_mean(image_size, batch_size)
        sd = self.get_RGB_sd(mean, image_size, batch_size)

        return {"mean": [mean[0].item(), mean[1].item(), mean[2].item()],
                "sd": [sd[0].item(), sd[1].item(), sd[2].item()]}


def print_img_infos_datasets():
    DATASET_FILE_PATHS_CONFIG = "dataset_file_args.json"
    HYPER_PARAMETER_CONFIG = "hyper_parameters.json"

    file_args = read_json_config(DATASET_FILE_PATHS_CONFIG)


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


def preprocess_text(text, remove_punctuation=True):
    """
    Removes punctuation and add lower case
    :param text:
    :return:
    """
    text = text.lower()
    text = word_tokenize(text)
    if remove_punctuation:
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
        return self.__class__.__name__ + f"(padding={0}, fill={1}, padding_mode={2})".format(self.padding, self.fill,
                                                                                             self.padding_mode)


def get_current_images_id(hparams, dataset_name):
    """
    Get the list of image ids corresponding to break_training_loop_percentage in hparams.
    It helps loading less captions, embeddings in create_list_of_captions_and_clean() , hence getting more samples per batches...
    :param hparams:
    :param dataset_name:
    :return:
    """

    train_file = hparams[dataset_name]
    image_dir = os.path.join(hparams['root'], train_file)
    caption_file_path = get_cleaned_captions_path(hparams, train_file)
    img_size = hparams["image_size"]
    transform_pipeline = transforms.Compose([CenteringPad(),
                                             transforms.Resize((img_size, img_size)),
                                             transforms.ToTensor()])

    coco_train_set = dset.CocoDetection(root=image_dir,
                                        annFile=caption_file_path,
                                        transform=transform_pipeline
                                        )
    train_loader = torch.utils.data.DataLoader(coco_train_set, batch_size=hparams["batch_size"])
    break_training_loop_percentage = hparams["break_training_loop_percentage"]
    break_training_loop_idx = max(int(len(train_loader) * break_training_loop_percentage / 100) - 1, 0)

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
    :param hparams:
    :param name:
    :return:
    """
    percentage = hparams["break_training_loop_percentage"]
    caption_dir = os.path.join(hparams['root'], "annotations")
    punct = ""
    if not remove_punctuation:
        punct = "punct_"
    use_reduced_set = False
    if percentage < 100:
        save_file_path = os.path.join(caption_dir, f"{percentage}_cleaned_captions_{punct + hparams[name]}.json")
        caption_file = Path(save_file_path)
        use_reduced_set = caption_file.is_file()
    if not use_reduced_set:
        save_file_path = os.path.join(caption_dir, f"cleaned_captions_{punct + hparams[name]}.json")
        caption_file = Path(save_file_path)
        file_available = caption_file.is_file()
    else:
        file_available = True
    if file_available:
        return save_file_path
    return None


def get_captions(hparams, name):
    """
    Only extracts the needed captions set in caption_number. Might reduce memory footprint for embeddings
    :param hparams:
    :param name:
    :return:
    """

    transform_pipeline, shuffle = model.CocoDatasetWrapper._get_transform_pipeline_and_shuffle(hparams, name)
    train_file = hparams[name]
    image_dir = os.path.join(hparams['root'], train_file)
    caption_file_path = get_correct_annotation_file(hparams, name)

    coco_train_set = dset.CocoDetection(root=image_dir,
                                        annFile=caption_file_path,
                                        transform=transform_pipeline
                                        )
    loader = torch.utils.data.DataLoader(coco_train_set, batch_size=hparams["batch_size"])
    captions_number = hparams["caption_number"]
    list = []
    for batch in tqdm(enumerate(loader)):
        captions = batch[1][1][:captions_number]
        list.extend([c for idx, caption_list in enumerate(captions) for c in caption_list["caption"]])
    return list

def create_list_of_captions_and_clean(hparams, name, img_list=None, remove_punctuation=True):
    """
    Given a caption json file for the COCO dataset, lower case the labels
    and add space before and after punctuation, Preprocessing function from
    "Natural Language with Pytorch", chapter 3.
    :param file_path:
    :param save_file_path:
    :return:
    """

    punct = ""
    if not remove_punctuation:
        punct = "punct_"
    file_path = get_cleaned_captions_path(hparams, punct + hparams[name])
    if not Path(file_path).is_file():
        file_path = get_captions_path(hparams, hparams[name])
    save_file_path = get_correct_annotation_file(hparams, name, remove_punctuation)
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
                    cleaned_caption = preprocess_text(caption["caption"], remove_punctuation)
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


def clean_caption_annotations(annotation_dir, annotation_list, remove_punctuation=True):
    for annotation in annotation_list:
        create_list_of_captions_and_clean(annotation_dir, annotation, None, remove_punctuation)


def calculate_rgb_stats():
    clean_caption_annotations("../data/annotations/", ["captions_train2017.json", "captions_val2017.json"])
    coco_train_set = dset.CocoDetection(root="../data/train2017",
                                        annFile="../data/annotations/cleaned_captions_train2017.json",
                                        transform=transforms.Compose([CenteringPad(),
                                                                      transforms.Resize((640, 640)),
                                                                      transforms.ToTensor()])
                                        )
    iss = ImageSizeStats(coco_train_set)
    t = torch.ones(3)

    rgb_means = iss.get_RGB_mean_sd()
    create_json_config(rgb_means, "rgb_stats.json")


def get_captions_path(hparams, dataset):
    return f"{hparams['root']}/annotations/captions_{dataset}.json"


def get_cleaned_captions_path(hparams, dataset):
    return f"{hparams['root']}/annotations/cleaned_captions_{dataset}.json"


def get_instance_path(hparams, dataset):
    return f"{hparams['root']}/annotations/instances_{dataset}.json"


def save_coco(file, info, licenses, images, annotations):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations}, coco, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def create_cocosplit(args):
    """
    #Downloaded from github.com/akarazniewicz/cocosplit.git@master and modified to just handle annotations
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

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=args.split, shuffle=True)
        if args.percentage < 0:
            args.percentage = 0
        if args.percentage > 100:
            args.percentage = 100
        break_x_idx = max(int(len(x) * args.percentage / 100) - 1, 0)
        break_y_idx = max(int(len(y) * args.percentage / 100) - 1, 0)
        save_coco(args.train, info, licenses, x[0:break_x_idx], filter_annotations(annotations, x[0:break_x_idx]))
        save_coco(args.test, info, licenses, y[0:break_y_idx], filter_annotations(annotations, y[0:break_y_idx]))

        print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


def reduce_cocosplit(args):
    """
    #Downloaded from github.com/akarazniewicz/cocosplit.git@master and modified to just handle annotations
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

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        if args.percentage < 0:
            args.percentage = 0.0
        if args.percentage > 100:
            args.percentage = 100.0

        x, _ = train_test_split(images, train_size=args.percentage / 100, shuffle=True)

        save_coco(args.train, info, licenses, x, filter_annotations(annotations, x))

        print("Saved")


def set_seed_everywhere(seed):
    """
    From NLP with Pytorch book.
    :param seed:
    :param cuda:
    :return:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_unpack_zip(zipurl, storage_directory):
    """
    From https://svaderia.github.io/articles/downloading-and-unzipping-a-zipfile/
    download the COCO images and put them under /data
    :param zipurl:
    :param storage_directory:
    :return:
    """
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


def unzip_glove():
    temp_path = "./data/glove.6B.zip"
    zf = ZipFile(temp_path)
    # Extract its contents into <extraction_path>
    # note that extractall will automatically create the path
    zf.extractall(path="./data")
    # close the ZipFile instance
    zf.close()
    os.remove(temp_path)


def reduce_annotation_size(annotation_directory="./data/annotations", cleaned_file_prefix="cleaned_captions_",
                           final_percentage=10):
    file_suffixes = ["train2017.json", "val2017.json", "test2017.json"]

    for fs in file_suffixes:
        arg_for_split = Namespace(annotations=f'{annotation_directory}/{cleaned_file_prefix + fs}',
                                  having_annotations=False,
                                  train=f'{annotation_directory}/{final_percentage}_{cleaned_file_prefix + fs}',
                                  percentage=final_percentage)
        reduce_cocosplit(arg_for_split)
        print("Annotations created:", f'{annotation_directory}/{final_percentage}_{cleaned_file_prefix + fs}')


if __name__ == '__main__':
    hparams = read_json_config("./hparams.json")

    # Because the original test labels are missing in the Coco dataset (remember, it was meant as a competition)
    # we need to split the traning dataset into training and testing 80% / 20%

    # arg_for_split = Namespace(annotations='../data/annotations/cleaned_captions_train2017.json', having_annotations=False, split=0.8,
    #         test='../data/annotations/cleaned_captions_test2017.json', train='../data/annotations/cleaned_captions_train2017.json', percentage=100)
    # create_cocosplit(arg_for_split)

    # clean_caption_annotations(hparams, ["train", "val"],  False)

    # percentage_list = [1, 2, 5, 6, 10]
    # for p in percentage_list:
    #     reduce_annotation_size("../data/annotations", "cleaned_captions_punct_", p)
    #     reduce_annotation_size("../data/annotations", "cleaned_captions_", 2)

    # for example from captions_train2017.json we get cleaned_captions_train2017.json and cleaned_captions_train2017_labels_only.json
    # clean_caption_annotations(hparams, ["train", "val", "test"])
    # download_unpack_zip(hparams["img_train_url"], hparams["root"])
    # download_unpack_zip(hparams["img_val_url"], hparams["root"])
    # download_unpack_zip(hparams["glove_url"], hparams["root"])
    # no zip unzip command on the potsdam server...
    unzip_glove()
