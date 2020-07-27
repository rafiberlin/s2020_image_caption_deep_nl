from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
import preprocessing as prep
import pandas as pd


class BleuScorer:

    @classmethod
    def evaluate_gold(cls, hparams, train_loader, idx_break=-1, prefix="train"):

        # NEVER do [{}]* 5!!!!
        # https://stackoverflow.com/questions/15835268/create-a-list-of-empty-dictionaries
        caption_number = hparams["caption_number"]
        if caption_number > 5 or caption_number is None:
            caption_number = 5
        hypothesis = [{} for _ in range(caption_number)]
        references = [{} for _ in range(caption_number)]
        v = train_loader.dataset.vectorizer
        gold_with_original = ""
        if hparams["gold_eval_with_original"]:
            gold_with_original = "_orig"

        for idx, current_batch in enumerate(tqdm(train_loader)):
            imgs, \
                annotations, _ = current_batch
            for sample_idx, image_id in enumerate(annotations[0]["image_id"]):
                # create the list of all 4 captions out of 5. Because range(5) is ordered, the result is
                # deterministic...
                for c in list(combinations(range(caption_number), caption_number - 1)):
                    for hypothesis_idx in range(caption_number):
                        if hypothesis_idx not in c:
                            # gold_eval_with_original with false will have reference captions with <UNK> token within
                            if hparams["gold_eval_with_original"]:
                                hypothesis[hypothesis_idx][image_id.item()] = [
                                    annotations[hypothesis_idx]["caption"][sample_idx]]
                                references[hypothesis_idx][image_id.item()] = [
                                    annotations[annotation_idx]["caption"][sample_idx] for annotation_idx in list(c)]
                            else:
                                hypothesis[hypothesis_idx][image_id.item()] = [
                                    v.decode(v.vectorize(annotations[hypothesis_idx]["caption"][sample_idx])[0])]
                                references[hypothesis_idx][image_id.item()] = [
                                    v.decode(v.vectorize(annotations[annotation_idx]["caption"][sample_idx])[0]) for
                                    annotation_idx in list(c)]
            if idx == idx_break:
                # useful for debugging
                break

        scores = []
        for ref, hyp in list(zip(references, hypothesis)):
            scores.append(cls.calc_scores(ref, hyp))

        pd_score = pd.DataFrame(scores).mean()

        if hparams["save_eval_results"]:
            dt = datetime.now(tz=None)
            timestamp = dt.strftime(hparams["timestamp_prefix"])
            filepath = os.path.join(hparams["model_storage"],
                                    timestamp + f"{prefix}_bleu_gold{gold_with_original}.json")
            prep.create_json_config(pd_score.to_dict(), filepath)

        return pd_score

    @classmethod
    def evaluate(cls, hparams, train_loader, network_model, end_token_idx=3, idx_break=-1, prefix="train"):
        # there is no other method to retrieve the current device on a model...
        device = next(network_model.parameters()).device
        hypothesis = {}
        references = {}
        v = train_loader.dataset.vectorizer
        caption_number = hparams["caption_number"]
        gold_with_original = ""
        if hparams["gold_eval_with_original"]:
            gold_with_original = "_orig"
        bw = ""
        sampler = None
        if hparams["sampling_method"] == "beam_search":
            beam_width = hparams["beam_width"]
            bw = f"_bw{hparams['beam_width']}"
            def sampler(x): return network_model.predict_beam(x, beam_width)
        elif hparams["sampling_method"] == "sample_search":
            bw = "_sc"

            def sampler(x): return network_model.predict_greedy_sample(
                x, end_token_idx)
        else:
            def sampler(x): return network_model.predict_greedy(
                x, end_token_idx)

        for idx, current_batch in enumerate(train_loader):
            imgs, annotations, _ = current_batch
            for sample_idx, image_id in tqdm(enumerate(annotations[0]["image_id"])):
                _id = image_id.item()
                starting_token = v.create_starting_sequence().to(device)
                img = imgs[sample_idx].unsqueeze(dim=0).to(device)
                caption = starting_token.unsqueeze(
                    dim=0).unsqueeze(dim=0).to(device)
                input_for_prediction = (img, caption)
                predicted_label = sampler(input_for_prediction)
                current_hypothesis = v.decode(predicted_label[0][0])
                hypothesis[_id] = [current_hypothesis]
                # with false, gold gaptions have <UNK> token
                if hparams["gold_eval_with_original"]:
                    # packs all 5 labels for one image with the corresponding image id
                    references[_id] = [annotations[annotation_idx]["caption"][sample_idx] for annotation_idx in
                                       range(caption_number)]
                else:
                    references[_id] = [v.decode(v.vectorize(annotations[annotation_idx]["caption"][sample_idx])[0]) for
                                       annotation_idx in
                                       range(caption_number)]
                if hparams["print_prediction"]:
                    print("\n#########################")
                    print("image", _id)
                    print("prediction", hypothesis[_id])
                    print("gold captions", references[_id])

            if idx == idx_break:
                # useful for debugging
                break
        score = cls.calc_scores(references, hypothesis)
        pd_score = pd.DataFrame([score])

        if hparams["save_eval_results"]:
            dt = datetime.now(tz=None)
            timestamp = dt.strftime(hparams["timestamp_prefix"])
            filepath = os.path.join(hparams["model_storage"],
                                    timestamp + f"{prefix}_bleu_prediction{bw}{gold_with_original}.json")
            filepath_2 = os.path.join(hparams["model_storage"],
                                      timestamp + f"{prefix}_bleu_prediction_scores{bw}{gold_with_original}.json")
            prep.create_json_config(
                {k: (hypothesis[k], references[k]) for k in hypothesis.keys()}, filepath)
            prep.create_json_config([score], filepath_2)

        """
        this code delivers the same results but we can't calculate a reasonable score on the 
        predefined labels to compare against gold (we always get 100%...)
        res looks like this : [{"image_id": 139, "caption": "a woman posing for the camera standing on skis"}, ... ]
        from pycocoevalcap.eval import COCOEvalCap
        coco_cap = COCO(caption_file_path)
        #imgIds = sorted([ batch_one[1][0]["image_id"][i].item() for i in range(batch_size)])
        imgIds = sorted([ id for id in hypothesis.keys()])
        res = [ {"image_id": k, "caption": hypothesis[k][0]} for k in sorted(hypothesis.keys())]
        prep.create_json_config(res, "./res_.json", 0)
        coco_res = coco_cap.loadRes("./res_.json")
        #CocoEvalBleuOnly is a copy Paste of the CocoEval class but where we only put BleuScorer...
        cocoEval = model.CocoEvalBleuOnly(coco_cap, coco_res)
        cocoEval.params['image_id'] = imgIds
        s = cocoEval.evaluate()
        """

        return pd_score

    @classmethod
    def calc_scores(cls, ref, hypo):
        """
        Code from https://www.programcreek.com/python/example/103421/pycocoevalcap.bleu.bleu.Bleu
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    @classmethod
    def perform_whole_evaluation(cls, hparams, loader, network, break_training_loop_idx=3, prefix="train"):
        print("##########################################################")
        print("\nRun complete evaluation for:", prefix)
        train_bleu_score = BleuScorer.evaluate(hparams, loader, network,
                                               idx_break=break_training_loop_idx, prefix=prefix)
        print("Unweighted Current Bleu Scores:\n", train_bleu_score)
        train_bleu_score_pd = train_bleu_score.to_numpy().reshape(-1)
        print("Weighted Current Bleu Scores:\n", train_bleu_score_pd.mean())
        print("Geometric Mean Current Bleu Score:\n", gmean(train_bleu_score_pd))
        print("\nRun complete evaluation for: gold")
        bleu_score_human_average = BleuScorer.evaluate_gold(hparams, loader, idx_break=break_training_loop_idx,
                                                            prefix=prefix)
        bleu_score_human_average_np = bleu_score_human_average.to_numpy().reshape(-1)
        print("Unweighted Gold Bleu Scores:\n", bleu_score_human_average)
        print("Weighted Gold Bleu Scores:\n",
              bleu_score_human_average_np.mean())
        print("Geometric Gold Bleu Scores:\n",
              gmean(bleu_score_human_average_np))
        print("##########################################################")
