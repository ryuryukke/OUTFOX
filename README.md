# OUTFOX🦊
### This is the official code and data for our new preprint: [OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples](https://arxiv.org/abs/2307.11729)

## 🔨 Setup
- python==3.9.2
```
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```
Run any script in the `scripts` directory.

`scripts/detection.sh` is a script for our OUTFOX detection and baseline detections, and `scripts/attacking.sh` is a script for our OUTFOX attacking and baseline paraphrasing attack.

## :page_facing_up: Dataset Info
We created our dataset based on [Kaggle FeedBack Prize](https://www.kaggle.com/competitions/feedback-prize-effectiveness) and contains 15,400 triplets of essay problem statements, human(native-student)-written essays, and LLM-generated essays. The native students range from 6th to 12th grade in the U.S.

We instruct three LMs to generate essays: ChatGPT(`gpt-3.5-turbo-0613`), GPT-3.5(`text-davinci-003`), and `FLAN-T5-XXL`.
We split the dataset into three parts: train/validation/test with 14400/500/500 examples, respectively.

This is supplemental information about the file names consisting of our dataset.

|File name|Content|
|---|------|
|`(train\|valid\|test)_problem_statements.pkl`| Essay problem statements in each set. |
|`(train\|valid\|test)_humans.pkl`| Human-written essays in each set. |
|`(train\|valid\|test)_lms.pkl`| LLM-generated essays in each set. |

Additionally, `(train\|valid\|test)_contexts.pkl` includes the prompts used to generate essays in each set. We use these to compute the likelihood in statistical outlier detectors.

We also provide the attacked essays by our OUTFOX attacker in `data/chatgpt/test/test_outfox_attacks.pkl` and the attacked essays by DIPPER in `data/dipper/(chatgpt|text_davinci_003|flan_t5_xxl)/test_attacks.pkl`.


## 📚 Citation
#### If you use any part of this work, please cite it as follows:
```
@misc{koike2023outfox,
      title={OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples}, 
      author={Ryuto Koike and Masahiro Kaneko and Naoaki Okazaki},
      year={2023},
      eprint={2307.11729},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```