import random
random.seed(42)
import openai
import os
from tqdm import tqdm
from utils.utils import load_pkl, save_pkl, generation_by_chatgpt, identify_attack_label, make_prompt_for_attack

openai.api_key = os.getenv("OPENAI_API_KEY")


def make_attacked_test(test_humans, test_problem_statements, test_ps_to_near_ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train):
    attacked_test = []
    
    for test_human, test_ps in tqdm(zip(test_humans, test_problem_statements), total=len(test_humans)):
        example_for_detection = test_ps_to_near_ps_human_lm_pairs_from_train[test_ps]
        lm_and_attack_labels = []
        for train_ps, _, train_lm in tqdm(example_for_detection):
            lm_and_attack_label = identify_attack_label(train_ps, train_lm, train_ps_to_near_ps_human_lm_pairs_from_train)
            lm_and_attack_labels.append(lm_and_attack_label)
        assert len(lm_and_attack_labels) == 10, 'invalid size of lm_and_attack_labels'
        random.shuffle(lm_and_attack_labels)
        prompt, human_essay_tokens = make_prompt_for_attack(lm_and_attack_labels, test_ps, test_human)
        attacked_essay = generation_by_chatgpt(prompt, human_essay_tokens)
        attacked_test.append(attacked_essay)
    
    return attacked_test


def main():
    test_humans = load_pkl('../../data/common/test/test_humans.pkl')
    test_problem_statements = load_pkl('../../data/common/test/test_problem_statements.pkl')

    # Loading top-k (problem statement, human-written essay, LLM-generated essay) sets retrieved in advance based on the problem statement, using tf-idf.
    # Whatever LMs (FLAN, GPT-3.5) to be detected, our OUTFOX detector consider the essays by ChatGPT.
    test_ps_to_near_ps_human_lm_pairs_from_train = load_pkl(f'../../data/chatgpt/util/test_ps_to_near_ps_human_lm_pairs_from_train.pkl')
    train_ps_to_near_ps_human_lm_pairs_from_train = load_pkl(f'../../data/chatgpt/util/train_ps_to_near_ps_human_lm_pairs_from_train.pkl')

    attacked_test = make_attacked_test(test_humans, test_problem_statements, test_ps_to_near_ps_human_lm_pairs_from_train, train_ps_to_near_ps_human_lm_pairs_from_train)
    save_pkl(attacked_test, f'../../data/chatgpt/test/test_outfox_attacks.pkl')
    print('Complete!')


if __name__ == '__main__':
    main()
