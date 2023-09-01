import numpy as np
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import functools
import time
from torch.nn import CrossEntropyLoss
from utils.utils import json2dict, load_pkl, save_pkl


# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


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


def load_base_model_and_tokenizer(name):
    base_model = T5ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, offload_folder="offload")
    base_tokenizer = T5Tokenizer.from_pretrained(name)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer


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
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
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
def get_entropy(text, context):
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


def get_perturbation_results_(span_length=10, n_perturbations=1, n_samples=500):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    all_texts = list(map(lambda x: x[0], [data for data in all_data]))
    all_texts = [mask_tokenizer.decode(mask_tokenizer(text)['input_ids'][:510]) if len(mask_tokenizer.tokenize(text)) > 510 else text for text in all_texts]
    context_text = list(map(lambda x: x[2], [data for data in all_data]))

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_text = perturb_fn([x for x in all_texts for _ in range(n_perturbations)])
    
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_text = perturb_fn(p_text)
        except AssertionError:
            break

    assert (
        len(p_text) == len(all_texts) * n_perturbations
    ), f"Expected {len(all_texts) * n_perturbations} perturbed samples, got {len(p_text)}"

    for idx in range(len(all_texts)):
        results.append(
            {
                "text": all_texts[idx],
                'context': context_text[idx],
                "perturbed_texts": p_text[idx * n_perturbations : (idx + 1) * n_perturbations],
            }
        )

    save_pkl(results, '../../data/flan_t5_xxl/test/essay_test_perturbed_results.pkl')

    load_base_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_lls = get_lls(res["perturbed_texts"], res['context'])
        res["text_ll"] = get_ll(res["text"], res['context'])
        res["all_perturbed_lls"] = p_lls
        res["perturbed_ll"] = np.mean(p_lls)
        res["perturbed_ll_std"] = np.std(p_lls) if len(p_lls) > 1 else 1

    return results


def compute_real_metrics(pred_scores, labels, detect_method_name):
    preds = []
    threshold = fixed_threshold[detect_method_name]

    preds = ['1' if pred_score >= threshold else '0' for pred_score in pred_scores]
    
    def compute_three_recalls(labels, preds):
        all_n, all_p, tn, tp = 0, 0, 0, 0
        for label, pred in zip(labels, preds):
            if label == '0':
                all_p += 1
            if label == '1':
                all_n += 1
            if label == pred == '0':
                tp += 1 
            if label == pred == '1':
                tn += 1
        human_rec, machine_rec = tp * 100 / all_p, tn * 100 / all_n
        avg_rec = (human_rec + machine_rec) / 2
        return (human_rec, machine_rec, avg_rec)
    
    human_rec, machine_rec, avg_rec = compute_three_recalls(labels, preds)
    acc, precision, recall, f1 = accuracy_score(labels, preds), precision_score(labels, preds, pos_label='1'), recall_score(labels, preds, pos_label='1'), f1_score(labels, preds, pos_label='1')
    return (human_rec, machine_rec, avg_rec, acc, precision, recall, f1)


def run_perturbation_experiment_(results, criterion):
    # compute diffs with perturbed
    predictions = []
    for res in results:
        if criterion == "d":
            predictions.append(res["text_ll"] - res["perturbed_ll"])
        elif criterion == "z":
            if res["perturbed_ll_std"] == 0:
                res["perturbed_ll_std"] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_texts']))}")
                print(f"Original text: {res['text']}")
            predictions.append(
                (res["text_ll"] - res["perturbed_ll"]) / res["perturbed_ll_std"]
            )

    labels = list(map(lambda x: x[1], all_data))

    human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_real_metrics(predictions, labels, 'detectgpt')
    
    print(f'Model: detectgpt')
    print(f"HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")


def run_baseline_threshold_experiment_(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm.tqdm(range(n_samples * 2 // batch_size), desc=f"Computing {name} criterion"):
        mixed_texts = list(map(lambda x: x[0], all_data[batch * batch_size : (batch + 1) * batch_size]))
        
        context_text = list(map(lambda x: x[2], all_data[batch * batch_size : (batch + 1) * batch_size]))
        
        for idx in range(len(mixed_texts)):
            results.append(
                {
                    "text": mixed_texts[idx],
                    "text_score": -criterion_fn(mixed_texts[idx], context_text[idx], log=True if 'log' in name else False) if 'rank' in name else criterion_fn(mixed_texts[idx], context_text[idx]),
                }
            )

    # compute prediction scores for real/sampled passages
    predictions = {
        "text_scores": [x["text_score"] for x in results],
    }

    labels = list(map(lambda x: x[1], all_data))

    human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_real_metrics(predictions['text_scores'], labels, name)
    
    print(f'Model: {name}')
    print(f"HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")


def mixed_human_lm_dataset(model_name):
    random.seed(42)
    human_texts = load_pkl('../../data/common/test/test_humans.pkl')
    # When you evaluate detectors on attacked texts, please specify a path to attacked texts.
    lm_texts = load_pkl(f'../../data/{model_name}/test/test_lms.pkl')
    contexts = load_pkl(f'../../data/common/test/test_contexts.pkl')
    
    human_texts_with_label, lm_texts_with_label = [(human_text, '0', context) for human_text, context in zip(human_texts, contexts)], [(lm_text, '1', context) for lm_text, context in zip(lm_texts, contexts)]
    all_texts_with_label = human_texts_with_label + lm_texts_with_label
    random.shuffle(all_texts_with_label)

    return all_texts_with_label


def eval_supervised_(model):
    mixed_data = list(map(lambda x: x[0], all_data))

    print(f"Beginning supervised evaluation with {model}...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    pred_scores = []

    with torch.no_grad():
        for batch in tqdm.tqdm(range(len(mixed_data) // batch_size), desc="Evaluating mixed_data"):
            batch_mixed_data = mixed_data[batch * batch_size : (batch + 1) * batch_size]
            batch_mixed_data = tokenizer(batch_mixed_data, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            pred_scores.extend(detector(**batch_mixed_data).logits.softmax(-1)[:, 0].tolist())

    labels = list(map(lambda x: x[1], all_data))

    human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_real_metrics(pred_scores, labels, model)
    
    print(f'Model: {model}')
    print(f"HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")


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

    if args.base_model_name == 'flan_t5_xxl':
        # mask filling t5 model
        print(f"Loading mask filling model {mask_filling_model_name}...")
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name)
        n_positions = mask_model.config.n_positions
        preproc_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions)
        print(f"Loading essay generation model {args.base_model_name}...")
        base_model, base_tokenizer = load_base_model_and_tokenizer('google/flan-t5-xxl')
        load_base_model()
    
    print(f"Loading a test test of {args.base_model_name}...")
    all_data = mixed_human_lm_dataset(args.base_model_name)
    fixed_threshold = json2dict(f'../../config/{args.base_model_name}_essay_threshold_config.json')

    if args.base_model_name == 'flan_t5_xxl':
        print('Starting baseline inference of statistical outlier approaches on a test set...')
        run_baseline_threshold_experiment_(get_ll, "likelihood", n_samples=sample_num)
        run_baseline_threshold_experiment_(get_rank, "rank", n_samples=sample_num)
        run_baseline_threshold_experiment_(get_rank, "logrank", n_samples=sample_num)
        run_baseline_threshold_experiment_(get_entropy, "entropy", n_samples=sample_num)
        # DetectGPT
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results_(args.span_length, n_perturbations)
            for perturbation_mode in ["z"]:
                output = run_perturbation_experiment_(
                    perturbation_results,
                    perturbation_mode,
                )
    else:
        print('Starting baseline inference of supervised classifiers on a test set...')
        eval_supervised_(model="roberta-base-openai-detector")
        eval_supervised_(model="roberta-large-openai-detector")
        eval_supervised_(model="Hello-SimpleAI/chatgpt-detector-roberta")