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
from pathlib import Path
import json
import bleu
import util
import vocab as vocab

HYPER_PARAMETER_CONFIG = "./hparams.json"
GLOVE_SCRIPT = "./utils/glove_conv.py"
PADDING_WORD = "<MASK>"
BEGIN_WORD = "<BEGIN>"
SEED = 1


def get_stop_loop_indices(hparams, train_loader, val_loader, test_loader):
    """
    Returns the indices needed to stop the loop early (only for debugging)
    Make sure that hparams["shuffle"] is false when debugging.
    Otherwise hparams["shuffle"] must always be true for correct learning
    :param hparams:
    :param train_loader:
    :param val_loader:
    :param test_loader:
    :return:
    """

    # Set "break_training_loop_percentage" to 100 in hparams.json to train on everything...
    if hparams["debug"]:
        break_training_loop_percentage = hparams["break_training_loop_percentage"]
        break_training_loop_idx = max(
            int(len(train_loader) * break_training_loop_percentage / 100) - 1, 0)
        break_val_loop_idx = max(
            int(len(val_loader) * break_training_loop_percentage / 100) - 1, 0)
        break_test_loop_idx = max(
            int(len(test_loader) * break_training_loop_percentage / 100) - 1, 0)
    else:
        break_training_loop_idx = len(train_loader)
        break_val_loop_idx = len(val_loader)
        break_test_loop_idx = len(test_loader)

    return break_training_loop_idx, break_val_loop_idx, break_test_loop_idx


def init_model(hparams, network, force_training=False):
    """
    Init the model with pre-existing learned values.
    Returns a boolean which indicates if the training should be started or not.
    :param hparams:
    :param network:
    :param force_training:
    :return:
    """

    # Generate output folder if non-existent
    model_dir = hparams["model_storage"]
    model_name = util.create_model_name(hparams)
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


def compute_loss_on_validation(val_loader, device, network):
    val_total_loss = torch.zeros(1, device=device)
    #val_loss_function = nn.CrossEntropyLoss().to(device)
    val_loss_function = nn.NLLLoss().to(device)
    with torch.no_grad():
        for val_idx, val_batch in enumerate(val_loader):
            val_images, val_in_captions, val_out_captions = val_batch
            del val_batch
            val_log_prediction = network(val_images.to(device), val_in_captions.to(device)).permute(0, 2, 1)
            val_out_captions = val_out_captions.to(device)
            val_total_loss += val_loss_function(val_log_prediction, val_out_captions)
            del val_images
            del val_in_captions
            del val_out_captions
    return val_total_loss.item()


def train(hparams, loss_function, network, train_loader, device, break_training_loop_idx, val_loader):
    """
    Performs the main training loop
    :param hparams:
    :param loss_function:
    :param network:
    :param train_loader:
    :param device:
    :param break_training_loop_idx:
    :param val_loader:
    :return:
    """

    model_dir = hparams["model_storage"]
    model_name = util.create_model_name(hparams)
    model_path = os.path.join(model_dir, model_name)

    if hparams["sgd_momentum"]:
        optimizer = optim.SGD(params=network.parameters(), momentum=hparams["sgd_momentum"], lr=hparams['lr'],
                              nesterov=True, weight_decay=hparams['weight_decay'])
    else:
        optimizer = optim.Adam(params=network.parameters(
        ), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

    start = timer()
    tb = None
    if hparams["use_tensorboard"]:
        tb = SummaryWriter(model_name)
        batch = next(iter(train_loader))
        grid = make_grid(batch[0])
        tb.add_image("images", grid)
        images, in_captions, out_captions = util.CocoDatasetWrapper.transform_batch_for_training(
            batch, device)
        tb.add_graph(network, (images, in_captions))

    # --- training loop ---
    network.train()
    scalar_total_loss = 0
    previous_val_loss = float('inf')
    previous_total_loss = float('inf')
    break_early_val_loss = False
    break_early = False
    temp_val_model = os.path.join(model_dir, f"val_loss_{model_name}")
    temp_model = os.path.join(model_dir, f"temp_{model_name}")
    total_val_loss_tracking = []
    total_loss_tracking = []
    for epoch in tqdm(range(hparams["num_epochs"])):
        total_loss = torch.zeros(1, device=device)
        for idx, current_batch in enumerate(train_loader):
            images, in_captions, out_captions = current_batch

            del current_batch
            optimizer.zero_grad()
            # The input for the NLL Loss is batch times number of classes (vocabulary in our case)
            # times sequence length => hence the permutation
            log_prediction = network(images.to(device), in_captions.to(device)).permute(0, 2, 1)
            out_captions = out_captions.to(device)
            loss = loss_function(log_prediction, out_captions)
            total_loss += loss
            loss.backward()

            # Should be helpful if we get NaN loss
            if hparams["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), hparams["clip_grad"])
            # Use optimizer to take gradient step
            optimizer.step()
            # hopefully helping for garbage collection an freeing up ram more quickly for GPU
            del images
            del in_captions
            del out_captions
            del log_prediction
            del loss
            # for dev purposes only
            if idx == break_training_loop_idx:
                break
        if (epoch + 1) % hparams["training_report_frequency"] == 0:
            scalar_total_loss = total_loss.item()
            print("\nTotal Loss:", scalar_total_loss, "Epoch:", epoch + 1)
            # Can be used to make graphs...
            total_loss_tracking.append((epoch + 1, scalar_total_loss))
            if hparams["compute_val_loss"]:
                val_loss = compute_loss_on_validation(
                    val_loader, device, network)
                print("Total Validation Loss:", val_loss,
                      "Epoch:", epoch + 1, "\n")
                # Can be used to make graphs..
                total_val_loss_tracking.append((epoch + 1, val_loss))
                if hparams["keep_best_val_loss"]:
                    if val_loss >= previous_val_loss:
                        print(
                            "Total Validation Loss got worse, reload last temporary model, break learning loop.")
                        break_early_val_loss = True
                        break
                    else:
                        torch.save(network.state_dict(), temp_val_model)
                previous_val_loss = val_loss
            if hparams["keep_best_total_loss"]:
                if scalar_total_loss > previous_total_loss:
                    print(
                        "Total Loss got worse, reload last temporary model, break learning loop.")
                    break_early = True
                    break
                else:
                    torch.save(network.state_dict(), temp_model)
                    previous_total_loss = scalar_total_loss
            if hparams["save_pending_model"]:
                temp_model = os.path.join(
                    model_dir, f"epoch_{str(epoch + 1)}_{model_name}")
                torch.save(network.state_dict(), temp_model)
            if hparams["use_tensorboard"]:
                tb.add_scalar("Total Loss", scalar_total_loss, epoch + 1)
        del total_loss
    end = timer()
    print("Overall Learning Time", end - start)
    if break_early_val_loss:
        print("Reload last good model before the validation got worse")
        network.load_state_dict(torch.load(temp_val_model))
        print("Best Validation loss:", previous_val_loss)
        os.remove(Path(temp_val_model))
    if break_early and not break_early_val_loss:
        print("Reload last good model before the total loss got worse")
        network.load_state_dict(torch.load(temp_model))
        print("Best Total loss:", previous_total_loss)
        os.remove(Path(temp_model))
    else:
        print("Total Loss", scalar_total_loss)

    if total_val_loss_tracking:
        path = os.path.join(model_dir, f"val_loss_{model_name}.json")
        with open(path, "w") as f:
            json.dump(total_val_loss_tracking, f)
        print("Saved Val Loss Stats", path)
    if total_loss_tracking:
        path = os.path.join(model_dir, f"loss_{model_name}.json")
        with open(path, "w") as f:
            json.dump(total_loss_tracking, f)
        print("Saved Loss Stats", path)

    torch.save(network.state_dict(), model_path)
    if tb:
        tb.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", help="hparams config file")
    parser.add_argument("--train", action="store_true", help="force training")
    parser.add_argument("--download", action="store_true",
                        help="download dataset")
    parser.add_argument("--prep", action="store_true",
                        help="preprocess dataset")
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

    if args.prep:
        prep.clean_caption_annotations(hparams, ["train", "val"])

    # Make sure the Cuda Start is fresh...
    torch.cuda.empty_cache()

    trainset_name = "train"
    # trainset_name = "val"
    valset_name = "val"
    testset_name = "test"
    device = hparams["device"]
    if not torch.cuda.is_available():
        print("Warning, only CPU processing available!")
        device = "cpu"
    else:
        print("CUDA GPU is available", "Number of machines:",
              torch.cuda.device_count())

    prep.set_seed_everywhere(SEED)
    cleaned_captions, train_annotation_ids = prep.get_captions(hparams, trainset_name)
    _, val_annotation_ids = prep.get_captions(hparams, valset_name)
    cutoff_for_unknown_words = hparams["cutoff"]

    c_vectorizer = vocab.CaptionVectorizer.from_dataframe(
        cleaned_captions, cutoff_for_unknown_words)
    padding_idx = None

    if (hparams["use_padding_idx"]):
        padding_idx = c_vectorizer.get_vocab()._token_to_idx[PADDING_WORD]

    annotation_train_loader = util.CocoDatasetAnnotation.create_dataloader(hparams, c_vectorizer, dataset_name=trainset_name, annotation_ids=train_annotation_ids)
    #TODO use the other Dataloader...
    annotation_val_loader = util.CocoDatasetAnnotation.create_dataloader(hparams, c_vectorizer, dataset_name=valset_name, annotation_ids=val_annotation_ids)
    #annotation_test_loader = util.CocoDatasetAnnotation.create_dataloader(hparams, c_vectorizer, dataset_name=testset_name, annotation_ids=train_annotation_ids)
    embedding = util.create_embedding(hparams, c_vectorizer, padding_idx)
    image_train_loader = util.CocoDatasetWrapper.create_dataloader(
        hparams, c_vectorizer, trainset_name)
    # The last parameter is needed, because the images of the testing set ar in the same directory as the images of the training set
    image_test_loader = util.CocoDatasetWrapper.create_dataloader(
        hparams, c_vectorizer, testset_name, "train2017")
    image_val_loader = util.CocoDatasetWrapper.create_dataloader(
        hparams, c_vectorizer, valset_name)

    network = model.RNNModel(hparams["hidden_dim"], pretrained_embeddings=embedding,
                             cnn_model=hparams["cnn_model"], rnn_layers=hparams["rnn_layers"],
                             rnn_model=hparams["rnn_model"], drop_out_prob=hparams["drop_out_prob"],
                             improve_cnn=hparams["improve_cnn"], teacher_forcing=hparams["teacher_forcing"]).to(device)

    start_training = init_model(hparams, network, args.train)
    break_training_loop_idx, break_val_loop_idx, break_test_loop_idx = get_stop_loop_indices(hparams, annotation_train_loader,
                                                                                             image_val_loader, image_test_loader)

    if start_training:
        #loss_function = nn.CrossEntropyLoss().to(device)
        loss_function = nn.NLLLoss().to(device)
        train(hparams, loss_function, network, annotation_train_loader,
              device, break_training_loop_idx, annotation_val_loader)

    bleu.BleuScorer.perform_whole_evaluation(
        hparams, image_train_loader, network, break_training_loop_idx, "train")
    bleu.BleuScorer.perform_whole_evaluation(
        hparams, image_val_loader, network, break_val_loop_idx, "val")
    #model.BleuScorer.perform_whole_evaluation(hparams, image_test_loader, network, break_test_loop_idx, "test")


if __name__ == '__main__':
    main()
