#!/usr/bin/env python3
from tqdm import tqdm
import torch
import torch.utils.data
import os
from torchvision.utils import make_grid
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
import model
import preprocessing as prep
import argparse
from torch.utils.tensorboard import SummaryWriter

HYPER_PARAMETER_CONFIG = "./hparams.json"
GLOVE_SCRIPT = "./util/glove_conv.py"
PADDING_WORD = "<MASK>"
BEGIN_WORD = "<BEGIN>"
SEED = 1


def init_model(hparams, network, force_training=False):
    """
    Init the model with pre-existing learned values.
    Returns a boolean which indicates if the training should be started or not.
    :param hparams:
    :param network:
    :param force_training:
    :return:
    """

    ## Generate output folder if non-existent
    model_dir = hparams["model_storage"]
    model_name = model.create_model_name(hparams)
    if not os.path.isdir(model_dir):
        try:
            os.mkdir(model_dir)
        except OSError:
            print(f"Creation of the directory {model_dir} failed")
    model_path = os.path.join(model_dir, model_name)
    print("Model save path:", model_path)

    start_training = True
    if os.path.isfile(model_path) and not force_training:
        network.load_state_dict(torch.load(model_path))
        start_training = False
        print("Skip Training")
    else:
        print("Start Training")
        # last_saved_model is either null in the Json file or contains the name of the pending model file to be loaded
        if hparams["last_saved_model"]:
            last_model = os.path.join(model_dir, hparams["last_saved_model"])
            if os.path.isfile(last_model):
                print("Load temporary model: ", last_model)
                network.load_state_dict(torch.load(last_model))
    return start_training


def train(hparams, loss_function, network, train_loader, device, break_training_loop_idx):
    """
    Performs the main training loop
    :param hparams:
    :param loss_function:
    :param network:
    :param train_loader:
    :param device:
    :param break_training_loop_idx:
    :return:
    """

    model_dir = hparams["model_storage"]
    model_name = model.create_model_name(hparams)
    model_path = os.path.join(model_dir, model_name)

    if hparams["sgd_momentum"]:
        optimizer = optim.SGD(params=network.parameters(), momentum=hparams["sgd_momentum"], lr=hparams['lr'],
                              nesterov=True)
    else:
        optimizer = optim.Adam(params=network.parameters(), lr=hparams['lr'])

    start = timer()
    tb = None
    if hparams["use_tensorboard"]:
        tb = SummaryWriter(model_name)
        batch = next(iter(train_loader))
        grid = make_grid(batch[0])
        tb.add_image("images", grid)
        images, in_captions, out_captions = model.CocoDatasetWrapper.transform_batch_for_training(batch, device)
        tb.add_graph(network, (images, in_captions))

    # --- training loop ---
    network.train()
    torch.cuda.empty_cache()
    scalar_total_loss = 0
    for epoch in tqdm(range(hparams["num_epochs"])):
        total_loss = torch.zeros(1).to(device)
        for idx, current_batch in enumerate(train_loader):
            images, in_captions, out_captions = model.CocoDatasetWrapper.transform_batch_for_training(current_batch,
                                                                                                      device)
            optimizer.zero_grad()
            # flatten all caption , flatten all batch and sequences, to make its category comparable
            # for the loss function
            out_captions = out_captions.reshape(-1)
            log_prediction = network(images, in_captions).reshape(out_captions.shape[0], -1)
            # Warning if we are unable to learn, use the contiguous function of the tensor
            # it insures that the sequence is not messed up during reshape
            loss = loss_function(log_prediction, out_captions)
            total_loss += loss
            loss.backward()
            # Use optimizer to take gradient step
            optimizer.step()
            # for dev purposes only
            if idx == break_training_loop_idx:
                break
            # hopefully helping for garbage collection an freeing up ram more quickly for GPU
            del current_batch
            del images
            del in_captions
            del out_captions
        if (epoch + 1) % hparams["training_report_frequency"] == 0:
            scalar_total_loss = total_loss.item()
            print("Total Loss:", scalar_total_loss, "Epoch:", epoch + 1)
            if hparams["save_pending_model"]:
                temp_model = os.path.join(model_dir, f"epoch_{str(epoch + 1)}_{model_name}")
                torch.save(network.state_dict(), temp_model)
            if hparams["use_tensorboard"]:
                tb.add_scalar("Total Loss", scalar_total_loss, epoch + 1)
        del total_loss
    end = timer()
    print("Overall Learning Time", end - start)
    print("Total Loss", scalar_total_loss)
    torch.save(network.state_dict(), model_path)
    if tb:
        tb.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", help="hparams config file")
    parser.add_argument("--train", action="store_true", help="force training")
    parser.add_argument("--download", action="store_true", help="download dataset")
    args = parser.parse_args()

    if args.params:
        hparams = prep.read_json_config(args.params)
    else:
        hparams = prep.read_json_config(HYPER_PARAMETER_CONFIG)

    if args.download:
        prep.download_unpack_zip(hparams["img_train_url"], hparams["root"])
        prep.download_unpack_zip(hparams["img_val_url"], hparams["root"])
        prep.download_unpack_zip(hparams["glove_url"], hparams["root"])
        with open(GLOVE_SCRIPT) as script_file:
            exec(script_file.read())
    #trainset_name = "val"
    trainset_name = "train"
    valset_name = "val"
    testset_name = "test"
    device = hparams["device"]
    if not torch.cuda.is_available():
        print("Warning, only CPU processing available!")
        device = "cpu"
    else:
        print("CUDA GPU is available", "Number of machines:", torch.cuda.device_count())
    prep.set_seed_everywhere(SEED)
    cleaned_captions = prep.create_list_of_captions_and_clean(hparams, trainset_name)
    c_vectorizer = model.CaptionVectorizer.from_dataframe(cleaned_captions)
    padding_idx = None
    if (hparams["use_padding_idx"]):
        padding_idx = c_vectorizer.get_vocab()._token_to_idx[PADDING_WORD]

    embedding = model.create_embedding(hparams, c_vectorizer, padding_idx)

    train_loader = model.CocoDatasetWrapper.create_dataloader(hparams, c_vectorizer, trainset_name)
    # The last parameter is needed, because the images of the testing set ar in the same directory as the images of the training set
    #test_loader = model.CocoDatasetWrapper.create_dataloader(hparams, c_vectorizer, testset_name, "val2017")
    #val_loader = model.CocoDatasetWrapper.create_dataloader(hparams, c_vectorizer, valset_name)

    network = model.RNNModel(hparams["hidden_dim"], pretrained_embeddings=embedding,
                             cnn_model=hparams["cnn_model"], rnn_layers=hparams["rnn_layers"], rnn_model=hparams["rnn_model"]).to(device)

    start_training = init_model(hparams, network, args.train)
    # Set "break_training_loop_percentage" to 100 in hparams.json to train on everything...
    break_training_loop_percentage = hparams["break_training_loop_percentage"]
    break_training_loop_idx = max(int(len(train_loader) * break_training_loop_percentage / 100) - 1, 0)
    #break_val_loop_idx = max(int(len(val_loader)*break_training_loop_percentage/100) - 1, 0)
    # break_test_loop_idx = max(int(len(test_loader)*break_training_loop_percentage/100) - 1, 0)
    if start_training:
        loss_function = nn.NLLLoss().to(device)
        train(hparams, loss_function, network, train_loader, device, break_training_loop_idx)
    model.BleuScorer.perform_whole_evaluation(hparams, train_loader, network, c_vectorizer, break_training_loop_idx, "train")
    #model.BleuScorer.perform_whole_evaluation(hparams, val_loader, network, c_vectorizer, break_training_loop_idx, "val")
    # model.BleuScorer.perform_whole_evaluation(hparams, test_loader, network, c_vectorizer, break_training_loop_idx)


if __name__ == '__main__':
    main()
