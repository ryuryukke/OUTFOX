import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
import torch
from tqdm import tqdm
import argparse
from utils.utils import load_pkl, save_pkl


class DipperParaphraser(object):
    def __init__(self, cuda_num, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.cuda_num = cuda_num
        if verbose:
            print(f"{model} model loaded in {time.time() - time1} sec")
        self.model.cuda(cuda_num)
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=2, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt", max_length=4096, truncation=True)
            final_input = {k: v.cuda(self.cuda_num) for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dipper_attack')
    parser.add_argument('--model_name', help='Specify target llms to be detected.', required=True, choices=['chatgpt', 'flan_t5_xxl', 'text_davinci_003'])
    parser.add_argument('--cuda_num', type=int, default=0)

    args = parser.parse_args()

    model_name = args.model_name
    cuda_num = args.cuda_num

    dp = DipperParaphraser(cuda_num)
    input_texts = load_pkl(f'../data/{model_name}/test/test_lms.pkl')
    attacks_by_dipper = []
    for input_text in tqdm(input_texts, desc=f'DIPPER Paraphrasing...'):
        output_l60_o60_sample = dp.paraphrase(input_text, lex_diversity=60, order_diversity=60, prefix='', do_sample=True, top_p=0.75, top_k=None, max_length=4096)
        attacks_by_dipper.append(output_l60_o60_sample)
    save_pkl(attacks_by_dipper, f'../data/dipper/{model_name}/test_attacks.pkl')