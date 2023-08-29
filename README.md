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

`scripts/detection.sh` is a script for our OUTFOX detection and baseline detections.

`scripts/attacking.sh` is a script for our OUTFOX attacking and baseline paraphrasing attack.

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
