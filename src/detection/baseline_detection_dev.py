import numpy as np
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
import functools
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from utils.utils import load_pkl, save_pkl


# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
    texta = " ".join(texta.split(" ")[:shorter_length])
    textb = " ".join(textb.split(" ")[:shorter_length])
    return texta, textb


def load_base_model_and_tokenizer(name):
    base_model = T5ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, offload_folder="offload")
    base_tokenizer = T5Tokenizer.from_pretrained(name)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer


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
    device = base_model.device
    with torch.no_grad():
        text = context + text
        if len(preproc_tokenizer.tokenize(text)) > 510:
            text = preproc_tokenizer.decode(preproc_tokenizer(text)['input_ids'][:510])
        # only for flan-t5-xxl
        contexts = base_tokenizer.tokenize(context)
        input_ids = base_tokenizer.encode(text, return_tensors="pt").to(device)
        logits = base_model(input_ids=input_ids, labels=input_ids).logits
        logits = logits.view(-1, logits.shape[-1])[len(contexts) - 1: -1, :]
        labels = input_ids.view(-1)[len(contexts):]
        
        loss_fn = CrossEntropyLoss()
        return -float(loss_fn(logits, labels))


def get_lls(texts, context):
    if context:
        return [get_ll(text, context) for text in texts]
    else:
        return [get_ll(text) for text in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, context, log=False):
    device = base_model.device
    with torch.no_grad():
        text = context + text
        if len(preproc_tokenizer.tokenize(text)) > 510:
            text = preproc_tokenizer.decode(preproc_tokenizer(text)['input_ids'][:510])
        
        # only for flan-t5-xxl
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
        
        # only for flan-t5-xxl
        contexts = base_tokenizer.tokenize(context)
        input_ids = base_tokenizer.encode(text, return_tensors="pt").to(device)
        logits = base_model(input_ids=input_ids, labels=input_ids).logits[:, len(contexts) - 1:-1]

        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_threshold_at_fpr_rate(fprs, thresholds, fpr_rate=0.01):
    idx = np.abs(fprs - fpr_rate).argmin()
    threshold = thresholds[idx]
    return threshold


def get_roc_metrics(real_preds, lm_preds):
    fprs, tprs, thresholds = roc_curve([0] * len(real_preds) + [1] * len(lm_preds), real_preds + lm_preds)
    roc_auc = auc(fprs, tprs)
    # get threhosld by Youden index
    youden_index = tprs - fprs
    # Find the threshold that maximizes the Youden Index
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]

    # threshold = get_threshold_at_fpr_rate(fprs, thresholds)
    return fprs.tolist(), tprs.tolist(), float(roc_auc), float(best_threshold)


def get_precision_recall_metrics(real_preds, lm_preds):
    precision, recall, _ = precision_recall_curve(
        [0] * len(real_preds) + [1] * len(lm_preds), real_preds + lm_preds
    )
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def get_perturbation_results(span_length=2, n_perturbations=100):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    human_text = [mask_tokenizer.decode(mask_tokenizer(text)['input_ids'][:510]) if len(mask_tokenizer.tokenize(text)) > 510 else text for text in data['human']]
    lm_text = [mask_tokenizer.decode(mask_tokenizer(text)['input_ids'][:510]) if len(mask_tokenizer.tokenize(text)) > 510 else text for text in data['lm']]
    context_text = data['context']

    perturb_fn = functools.partial(perturb_texts, span_length, pct=0.3)


    p_lm_text = perturb_fn([x for x in lm_text for _ in range(n_perturbations)])
    p_human_text = perturb_fn([x for x in human_text for _ in range(n_perturbations)])

    for _ in range(n_perturbation_rounds - 1):
        try:
            p_lm_text, p_human_text = perturb_fn(p_lm_text), perturb_fn(p_human_text)
        except AssertionError:
            break

    assert (
        len(p_lm_text) == len(lm_text) * n_perturbations
    ), f"Expected {len(lm_text) * n_perturbations} perturbed lms, got {len(p_lm_text)}"
    assert (
        len(p_human_text) == len(human_text) * n_perturbations
    ), f"Expected {len(human_text) * n_perturbations} perturbed lms, got {len(p_human_text)}"

    for idx in range(len(human_text)):
        results.append(
            {
                'human': human_text[idx],
                'lm': lm_text[idx],
                'context': context_text[idx],
                "perturbed_lm": p_lm_text[idx * n_perturbations : (idx + 1) * n_perturbations],
                "perturbed_human": p_human_text[idx * n_perturbations : (idx + 1) * n_perturbations],
            }
        )
    
    save_pkl(results, '../data/flan_t5_xxl/valid/valid_perturbed_results.pkl')

    load_base_model()

    for res in tqdm(results, desc="Computing log likelihoods"):
        p_lm_ll = get_lls(res["perturbed_lm"], res['context'])
        p_human_ll = get_lls(res["perturbed_human"], res['context'])
        res["human_ll"] = get_ll(res['human'], res['context'])
        res["lm_ll"] = get_ll(res['lm'], res['context'])
        res["all_perturbed_lm_ll"] = p_lm_ll
        res["all_perturbed_human_ll"] = p_human_ll
        res["perturbed_lm_ll"] = np.mean(p_lm_ll)
        res["perturbed_human_ll"] = np.mean(p_human_ll)
        res["perturbed_lm_ll_std"] = np.std(p_lm_ll) if len(p_lm_ll) > 1 else 1
        res["perturbed_human_ll_std"] = np.std(p_human_ll) if len(p_human_ll) > 1 else 1
    
    return results


def run_perturbation_experiment(results, criterion, n_perturbations=1):
    # compute diffs with perturbed
    predictions = {"real": [], "lms": []}
    for res in results:
        if criterion == "d":
            predictions["real"].append(res["human_ll"] - res["perturbed_human_ll"])
            predictions["lms"].append(res["lm_ll"] - res["perturbed_lm_ll"])
        elif criterion == "z":
            if res["perturbed_human_ll_std"] == 0:
                res["perturbed_human_ll_std"] = 1
                print("WARNING: std of perturbed human is 0, setting to 1")
                print(f"Number of unique perturbed human texts: {len(set(res['perturbed_human']))}")
                print(f"human text: {res['human']}")
            if res["perturbed_lm_ll_std"] == 0:
                res["perturbed_lm_ll_std"] = 1
                print("WARNING: std of perturbed lm is 0, setting to 1")
                print(f"Number of unique perturbed lm texts: {len(set(res['perturbed_lm']))}")
                print(f"lm text: {res['lm']}")
            predictions["real"].append(
                (res["human_ll"] - res["perturbed_human_ll"]) / res["perturbed_human_ll_std"]
            )
            predictions["lms"].append(
                (res["lm_ll"] - res["perturbed_lm_ll"]) / res["perturbed_lm_ll_std"]
            )

    _, _, roc_auc, threshold = get_roc_metrics(predictions["real"], predictions["lms"])
    _, _, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["lms"])
    name = f"perturbation_{n_perturbations}_{criterion}"
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    print(f'Model: {name} Threshold: {threshold}')


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        human_text = data['human'][batch * batch_size : (batch + 1) * batch_size]
        lm_text = data['lm'][batch * batch_size : (batch + 1) * batch_size]
        context_text = data['context'][batch * batch_size : (batch + 1) * batch_size]
        
        for idx in range(len(human_text)):
            results.append(
                {
                    'human': human_text[idx],
                    "human_crit": -criterion_fn(human_text[idx], context_text[idx], log=True if 'log' in name else False) if 'rank' in name else criterion_fn(human_text[idx], context_text[idx]),
                    'lm': lm_text[idx],
                    "lm_crit": -criterion_fn(lm_text[idx], context_text[idx], log=True if 'log' in name else False) if 'rank' in name else criterion_fn(lm_text[idx], context_text[idx]),
                }
            )

    # compute prediction scores for real/lm passages
    predictions = {
        "real": [x["human_crit"] for x in results],
        "lms": [x["lm_crit"] for x in results],
    }

    _, _, roc_auc, threshold = get_roc_metrics(predictions["real"], predictions["lms"])
    _, _, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["lms"])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    print(f'Model: {name} Threshold: {threshold}')


def eval_supervised(data, model):
    print(f"Beginning supervised evaluation with {model}...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    real, fake = data["human"], data["lm"]

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

    _, _, roc_auc, threshold = get_roc_metrics(real_preds, fake_preds)
    _, _, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    print(f'Model: {model} Threshold: {threshold}')

    # free GPU memory
    del detector
    torch.cuda.empty_cache()



if __name__ == "__main__":
    DEVICE = "cuda:0"

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
    parser.add_argument("--do_top_k", action="store_true")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--do_top_p", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--buffer_size", type=int, default=2)
    parser.add_argument("--mask_top_p", type=float, default=1.0)
    parser.add_argument("--pre_perturb_pct", type=float, default=0.0)
    parser.add_argument("--pre_perturb_span_length", type=int, default=5)
    
    args = parser.parse_args()

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    sample_num = args.sample_num
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds

    print(f"Loading mask filling model {mask_filling_model_name}...")
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions)

    if args.base_model_name == 'flan_t5_xxl':
        print(f"Loading essay generation model {args.base_model_name}...")
        base_model, base_tokenizer = load_base_model_and_tokenizer('google/flan-t5-xxl')
        load_base_model()

    print(f"Loading a dev test of {args.base_model_name}...")
    data = dict()
    data['lm'] = load_pkl(f'../data/{args.base_model_name}/valid/valid_lms.pkl')
    data['human'] = load_pkl('../data/common/valid/valid_humans.pkl')
    # 'context' is the instruction we used when generating essays.
    data['context'] = load_pkl(f'../data/{args.base_model_name}/valid/valid_contexts.pkl')
    
    humans, lms = data['human'][:], data['lm'][:]
    data['human'], data['lm'] = [], []
    for human, lm in zip(humans, lms):
        human, lms = trim_to_shorter_length(human, lms)
        data['human'].append(human)
        data['lm'].append(lm)

    if args.base_model_name == 'flan_t5_xxl':
        print('Starting baseline inference of statistical outlier approaches on a dev set...')
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
                    n_perturbations=n_perturbations,
                )
    else:
        print('Starting baseline inference of supervised classifiers on a dev set...')
        eval_supervised(data, model="roberta-base-openai-detector")
        eval_supervised(data, model="roberta-large-openai-detector")
        eval_supervised(data, model="Hello-SimpleAI/chatgpt-detector-roberta")