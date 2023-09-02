import numpy as np
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import pickle
import functools
import time
from torch.nn import CrossEntropyLoss
import pickle


# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_base_model():
    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    base_model.to(DEVICE)
    print(f"DONE ({time.time() - start:.2f}s)")


def load_mask_model():
    print("MOVING MASK MODEL TO GPU...", end="", flush=True)
    start = time.time()
    base_model.cpu()
    mask_model.to(DEVICE)
    print(f"DONE ({time.time() - start:.2f}s)")


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = " ".join(tokens)
    return text


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
    texta = " ".join(texta.split(" ")[:shorter_length])
    textb = " ".join(textb.split(" ")[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, "idx_occurrence must be > 0"
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def load_base_model_and_tokenizer(name):
    print(f"Loading BASE model {args.base_model_name}...") 
    base_model = T5ForConditionalGeneration.from_pretrained(name)
    base_tokenizer = T5Tokenizer.from_pretrained(name)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def get_roc_metrics(real_preds, sample_preds):
    fprs, tprs, thresholds = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fprs, tprs)
    # get threhosld by Youden index
    youden_index = tprs - fprs
    # Find the threshold that maximizes the Youden Index
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]

    return fprs.tolist(), tprs.tolist(), float(roc_auc), float(best_threshold)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve(
        [0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds
    )
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]

    n_expected = count_masks(masked_texts)

    tmp_tokens = tokens[:]
    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tmp_tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
            tokens[idx] = text
    
    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    
    while "" in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
        print(f"WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].")
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if "11b" in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i : i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


# Get the log likelihood of each text under the base_model
def get_ll(text, context):
    with torch.no_grad():
        text = context + text
        if len(preproc_tokenizer.tokenize(text)) > 510:
            text = preproc_tokenizer.decode(preproc_tokenizer(text)['input_ids'][:510])

        contexts = base_tokenizer.tokenize(context)
        input_ids = base_tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        logits = base_model(input_ids=input_ids, labels=input_ids).logits
        logits = logits.view(-1, logits.shape[-1])[len(contexts) - 1: -1, :]
        labels = input_ids.view(-1)[len(contexts):]

        loss_fn = CrossEntropyLoss()
        return -float(loss_fn(logits, labels))


def get_lls(texts, context):
    return [get_ll(text, context) for text in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, context, log=False):
    device = base_model.device
    with torch.no_grad():
        text = context + text
        if len(preproc_tokenizer.tokenize(text)) > 510:
            text = preproc_tokenizer.decode(preproc_tokenizer(text)['input_ids'][:510])
        
        contexts = base_tokenizer.tokenize(context)
        input_ids = base_tokenizer.encode(text, return_tensors="pt").to(device)
        logits = base_model(input_ids=input_ids, labels=input_ids).logits[:, len(contexts) - 1:-1]
        labels = input_ids[:, len(contexts):]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (
            timesteps == torch.arange(len(timesteps)).to(timesteps.device)
        ).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text, context=None):
    device = base_model.device
    with torch.no_grad():

        text = context + text
        if len(preproc_tokenizer.tokenize(text)) > 510:
            text = preproc_tokenizer.decode(preproc_tokenizer(text)['input_ids'][:510])
        
        contexts = base_tokenizer.tokenize(context)
        input_ids = base_tokenizer.encode(text, return_tensors="pt").to(device)
        logits = base_model(input_ids=input_ids, labels=input_ids).logits[:, len(contexts) - 1:-1]

        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_perturbation_results(span_length=2, n_perturbations=100):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = [mask_tokenizer.decode(mask_tokenizer(text)['input_ids'][:510]) if len(mask_tokenizer.tokenize(text)) > 510 else text for text in data["original"]]
    sampled_text = [mask_tokenizer.decode(mask_tokenizer(text)['input_ids'][:510]) if len(mask_tokenizer.tokenize(text)) > 510 else text for text in data["sampled"]]

    context_text = data['input']

    perturb_fn = functools.partial(perturb_texts, span_length=2, pct=0.3)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])

    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert (
        len(p_sampled_text) == len(sampled_text) * n_perturbations
    ), f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert (
        len(p_original_text) == len(original_text) * n_perturbations
    ), f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append(
            {
                "original": original_text[idx],
                "sampled": sampled_text[idx],
                'context': context_text[idx] if type_of_task != 'open-ended' else None,
                "perturbed_sampled": p_sampled_text[idx * n_perturbations : (idx + 1) * n_perturbations],
                "perturbed_original": p_original_text[idx * n_perturbations : (idx + 1) * n_perturbations],
            }
        )
    
    with open('../../data/flan_t5_xxl/valid/valid_perturbed_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    load_base_model()

    for res in tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"], res['context'])
        p_original_ll = get_lls(res["perturbed_original"], res['context'])
        res["original_ll"] = get_ll(res["original"], res['context'])
        res["sampled_ll"] = get_ll(res["sampled"], res['context'])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1
    
    return results


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {"real": [], "samples": []}
    for res in results:
        if criterion == "d":
            predictions["real"].append(res["original_ll"] - res["perturbed_original_ll"])
            predictions["samples"].append(res["sampled_ll"] - res["perturbed_sampled_ll"])
        elif criterion == "z":
            if res["perturbed_original_ll_std"] == 0:
                res["perturbed_original_ll_std"] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res["perturbed_sampled_ll_std"] == 0:
                res["perturbed_sampled_ll_std"] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions["real"].append(
                (res["original_ll"] - res["perturbed_original_ll"]) / res["perturbed_original_ll_std"]
            )
            predictions["samples"].append(
                (res["sampled_ll"] - res["perturbed_sampled_ll"]) / res["perturbed_sampled_ll_std"]
            )

    fpr, tpr, roc_auc, threshold = get_roc_metrics(predictions["real"], predictions["samples"])
    p, r, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["samples"])
    name = f"perturbation_{n_perturbations}_{criterion}"
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    print(f'Model: {name} Threshold: {threshold}')


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size : (batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size : (batch + 1) * batch_size]
        context_text = data["input"][batch * batch_size : (batch + 1) * batch_size]
        
        for idx in range(len(original_text)):
            results.append(
                {
                    "original": original_text[idx],
                    "original_crit": -criterion_fn(original_text[idx], context_text[idx], log=True if 'log' in name else False) if 'rank' in name else criterion_fn(original_text[idx], context_text[idx]),
                    "sampled": sampled_text[idx],
                    "sampled_crit": -criterion_fn(sampled_text[idx], context_text[idx], log=True if 'log' in name else False) if 'rank' in name else criterion_fn(sampled_text[idx], context_text[idx]),
                }
            )

    # compute prediction scores for real/sampled passages
    predictions = {
        "real": [x["original_crit"] for x in results],
        "samples": [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc, threshold = get_roc_metrics(predictions["real"], predictions["samples"])
    p, r, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["samples"])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    print(f'Model: {name} Threshold: {threshold}')


def eval_supervised(data, model):
    print(f"Beginning supervised evaluation with {model}...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    real, fake = data["original"], data["sampled"]

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size : (batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:, 0].tolist())

        fake_preds = []
        for batch in tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size : (batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:, 0].tolist())

    predictions = {
        "real": real_preds,
        "samples": fake_preds,
    }

    fpr, tpr, roc_auc, threshold = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    print(f'Model: {model} Threshold: {threshold}')

    # free GPU memory
    del detector
    torch.cuda.empty_cache()


if __name__ == "__main__":
    DEVICE = "cuda:2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pct_words_masked", type=float, default=0.3)  # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--n_perturbation_list", type=str, default="10")
    parser.add_argument("--n_perturbation_rounds", type=int, default=1)
    parser.add_argument("--base_model_name", type=str, default="flan_t5_xxl", choices=['chatgpt', 'flan_t5_xxl', 'text_davinci_003'])
    parser.add_argument("--mask_filling_model_name", type=str, default="t5-large")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--base_half", action="store_true")
    parser.add_argument("--do_top_k", action="store_true")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--do_top_p", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--buffer_size", type=int, default=2)
    parser.add_argument("--mask_top_p", type=float, default=1.0)
    parser.add_argument("--cache_dir", type=str, default="~/.cache")
    
    args = parser.parse_args()

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    sample_num = args.sample_num
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    if args.base_model_name == 'flan_t5_xxl':
        # essay generation model
        base_model, base_tokenizer = load_base_model_and_tokenizer('google/flan-t5-xxl')
        # mask filling t5 model
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f"Loading mask filling model {mask_filling_model_name}...")
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            mask_filling_model_name, **int8_kwargs, **half_kwargs)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512

        preproc_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions)

        load_base_model()

    ## Preparing essay data to be detected ##
    data = dict()
    with open(f'../../data/common/valid/valid_contexts.pkl', 'rb') as fa, open(f'../../data/{args.base_model_name}/valid/valid_lms.pkl', 'rb') as fb, open('../../data/common/valid/valid_humans.pkl', 'rb') as fc:
        data['input'] = pickle.load(fa)
        data['sampled'] = pickle.load(fb)
        data['original'] = pickle.load(fc)
    
    originals, samples = data['original'][:], data['sampled'][:]
    data['original'], data['sampled'] = [], []
    for original, sample in zip(originals, samples):
        original, sample = trim_to_shorter_length(original, sample)
        data['original'].append(original)
        data['sampled'].append(sample)
    ## Preparing essay data to be detected ##

    if args.base_model_name == 'flan_t5_xxl':
        # Statistical Outlier Approaches
        run_baseline_threshold_experiment(get_ll, "likelihood", n_samples=sample_num)
        run_baseline_threshold_experiment(get_rank, "rank", n_samples=sample_num)
        run_baseline_threshold_experiment(get_rank, "logrank", n_samples=sample_num)
        run_baseline_threshold_experiment(get_entropy, "entropy", n_samples=sample_num)
        # DetectGPT
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results(args.span_length, n_perturbations)
            for perturbation_mode in ["z"]:
                output = run_perturbation_experiment(
                    perturbation_results,
                    perturbation_mode,
                    span_length=args.span_length,
                    n_perturbations=n_perturbations,
                    n_samples=sample_num,
                )
    else:
        # Supervised Classifiers
        eval_supervised(data, model="roberta-base-openai-detector")
        eval_supervised(data, model="roberta-large-openai-detector")
        eval_supervised(data, model="Hello-SimpleAI/chatgpt-detector-roberta")

    


