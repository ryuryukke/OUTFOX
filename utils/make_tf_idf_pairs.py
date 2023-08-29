from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
from utils.utils import load_pkl, save_pkl


def get_test_ps_to_near_ps_human_lm_pairs_from_train_tf_idf(train_humans, train_lms, test_problem_statements, train_problem_statements):
    test_ps_to_near_lm_pss_from_train = dict()
    test_size = len(test_problem_statements)
    test_train_problem_statements = test_problem_statements + train_problem_statements

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(test_train_problem_statements)

    test_problem_statements_vectors, train_problem_statements_vectors = X[:test_size], X[test_size:]
    for i, test_ps_vector in tqdm(enumerate(test_problem_statements_vectors), total=test_size, desc='Collecting near human, lm: test->train'):
        test_ps_vector = np.squeeze(test_ps_vector.toarray())
        cos_sims = []
        for j, train_ps_vector in enumerate(train_problem_statements_vectors):
            train_ps_vector = np.squeeze(train_ps_vector.toarray())
            cos_sims.append((np.dot(test_ps_vector, train_ps_vector) / (np.linalg.norm(test_ps_vector) * np.linalg.norm(train_ps_vector)), j))
        max_sorted_cos_sims = sorted(cos_sims, key=lambda x: x[0], reverse=True)
        max_indices = list(map(lambda x: x[1], max_sorted_cos_sims[:10]))
        near_lm_pss = [(train_problem_statements[k], train_humans[k], train_lms[k]) for k in max_indices]
        test_ps_to_near_lm_pss_from_train[test_problem_statements[i]] = near_lm_pss
    return test_ps_to_near_lm_pss_from_train


def get_train_ps_to_near_ps_human_lm_pairs_from_train_tf_idf(train_humans, train_lms, train_problem_statements):
    train_ps_to_near_ps_human_lm_pairs_from_train = dict()

    vectorizer = TfidfVectorizer()
    train_problem_statements_vectors = vectorizer.fit_transform(train_problem_statements)

    for i, train_ps_vector in tqdm(enumerate(train_problem_statements_vectors), total=len(train_humans), desc='Collecting near human, lm: train->train'):
        train_ps_vector_target = np.squeeze(train_ps_vector.toarray())
        cos_sims = []
        for j, train_ps_vector in enumerate(train_problem_statements_vectors):
            train_ps_vector = np.squeeze(train_ps_vector.toarray())
            cos_sims.append((np.dot(train_ps_vector_target, train_ps_vector) / (np.linalg.norm(train_ps_vector_target) * np.linalg.norm(train_ps_vector)), j))
        max_sorted_cos_sims = sorted(cos_sims, key=lambda x: x[0], reverse=True)
        max_indices = list(map(lambda x: x[1], max_sorted_cos_sims[:11]))
        # to ignore the same problem statement, max_indices[1:]
        near_human_lm_pair = [(train_problem_statements[k], train_humans[k], train_lms[k]) for k in max_indices[1:]]
        train_ps_to_near_ps_human_lm_pairs_from_train[train_problem_statements[i]] = near_human_lm_pair
    return train_ps_to_near_ps_human_lm_pairs_from_train


def main():
    train_humans, train_lms = load_pkl(f'../data/common/train/train_humans.pkl'), load_pkl(f'../data/chatgpt/train/train_lms.pkl')
    test_problem_statements, train_problem_statements = load_pkl(f'../data/common/test/test_problem_statements.pkl'), load_pkl(f'../data/common/train/train_problem_statements.pkl')
    test_ps_to_near_ps_human_lm_pairs_from_train = get_test_ps_to_near_ps_human_lm_pairs_from_train_tf_idf(train_humans, train_lms, test_problem_statements, train_problem_statements)
    train_ps_to_near_ps_human_lm_pairs_from_train = get_train_ps_to_near_ps_human_lm_pairs_from_train_tf_idf(train_humans, train_lms, train_problem_statements)
    save_pkl(test_ps_to_near_ps_human_lm_pairs_from_train, f'../data/chatgpt/util/test_ps_to_near_ps_human_lm_pairs_from_train.pkl')
    save_pkl(train_ps_to_near_ps_human_lm_pairs_from_train, f'../data/chatgpt/util/train_ps_to_near_ps_human_lm_pairs_from_train.pkl')


if __name__ == '__main__':
    main()