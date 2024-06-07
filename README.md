# OUTFOX🦊
![](https://img.shields.io/badge/Made_with-python-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2307.11729-b31b1b.svg)](https://arxiv.org/abs/2307.11729)
[![LICENSE](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://github.com/ryuryukke/OUTFOX?tab=Apache-2.0-1-ov-file)

### This is the official code and data for our [AAAI 2024 paper](https://ojs.aaai.org/index.php/AAAI/article/view/30120), "OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples"

<p align="center">
  <img src="https://github.com/ryuryukke/OUTFOX/assets/61570900/4626abf6-5c75-43c9-91c0-812804e79104" width="500"/>
</p>

## 📖 Introduction
Current LLM-generated text detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts.
Furthermore, there is the unexplored risk where malicious users might exploit LLMs to create texts specifically designed to evade detection.

In this paper, we propose _**OUTFOX**_, a framework that improves the robustness of LLM detectors by allowing _**both the detector and the attacker to consider each other's output**_.
In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker.

Experiments in the domain of student essays show that...
- The proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points F1-score. 
- The proposed detector shows a _**state-of-the-art**_ detection performance: up to 96.9 points F1-score, beating existing detectors on non-attacked texts.
- The proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.

## 📢 Updates
- **Feb 2024**: Presented in AAAI 2024, Vancouver! [[poster](https://drive.google.com/file/d/1b4qm0wvCftNA2MKr5nevDtTALzUdqtbW/view?usp=drive_link)]
- **Aug 2023**: Our code and essay dataset are now available!

## 🔨 Setup
- python==3.9.2
```
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```
Run any script in the `scripts` directory.

`scripts/detection.sh` is a script for our OUTFOX detection and baseline detections.

`scripts/attacking.sh` is a script for our OUTFOX attacking and baseline paraphrasing attack.

## :page_facing_up: Dataset Info
We created our dataset based on [Kaggle FeedBack Prize](https://www.kaggle.com/competitions/feedback-prize-effectiveness), and our dataset contains 15,400 triplets of essay problem statements, human(native-student)-written essays, and LLM-generated essays. The native students range from 6th to 12th grade in the U.S.

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
#### If you find our code/data/models or ideas useful in your research, please consider citing our work as follows:
```
@InProceedings{Koike:OUTFOX:2024,
  author    = {Ryuto Koike and Masahiro Kaneko and Naoaki Okazaki},
  title     = {OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples},
  booktitle = {Proceedings of the 38th AAAI Conference on Artificial Intelligence},
  year      = {2024},
  month     = {February},
  address   = {Vancouver, Canada}
}
```

## 📩 Contact
Feel free to contact [Ryuto Koike](ryuto.koike@nlp.c.titech.ac.jp) if you have any questions.
