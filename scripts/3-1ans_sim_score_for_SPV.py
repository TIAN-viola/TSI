#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration

dataset_to_fileList = {
    'MOH-X':['MOH-X.csv'],
    'TroFi':['TroFi.csv'],

}
import pdb

def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='results/', type=str)
    parser.add_argument('--_model', default='gpt-3.5-turbo-0613', type=str, help='gpt-3.5-turbo-0613')
    parser.add_argument('--baseline', default='SPV-1que-1.0-extract', type=str, help='')
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results-similar-score/', type=str)
    args, unparsed = parser.parse_known_args()
    return args

class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.baseline, args._model, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.baseline, args._model, args.dataset)
    
    def get_dataset(self, file):
        file_dir = os.path.join(self.data_root_path, file)
        examples = []
        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)  # skip the headline
            for i, line in enumerate(lines):
                # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                examples.append(line)
        return examples

    def write_dataset(self, file, head, examples):
        if not os.path.exists(self.output_root_path):
            os.makedirs(self.output_root_path)
        with open(os.path.join(self.output_root_path, file), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            for line in examples:
                writer.writerow(line)

    def get_dataset_output(self, file):
        if not os.path.exists(os.path.join(self.output_root_path, file)):
            return []
        else:
            with open(os.path.join(self.output_root_path, file), 'r', encoding='utf-8') as f:
                lines = csv.reader(f)
                next(lines)  # skip the headline
                examples = []
                for i, line in enumerate(lines):
                    # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                    # example sentence may be empty, caution for processing
                    examples.append(line)
            return examples


def main(args):
    dataloader = Data_Processor(args)
    new_samples = []
    print(args.data_dir, args.baseline, args._model, args.dataset)
    tokenizer = T5Tokenizer.from_pretrained('./t5-3b')
    model = T5ForConditionalGeneration.from_pretrained('./t5-3b')
    model.cuda()
    for file in dataset_to_fileList[args.dataset]:
        samples = dataloader.get_dataset(file)
        for sample in samples:
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_kdg_text, answer1_list = sample
            answer1_list = eval(answer1_list)
            answer2_list = eval(answer2_list)
            # answer1
            answer1_scores_list = []
            for answer1_1 in answer1_list:
                answer1_scores = []
                input_list = []
                for answer1_2 in answer1_list:
                    input_list.append("stsb sentence 1: "+answer1_1+" sentence 2: "+answer1_2)
                input_ids = tokenizer(input_list, return_tensors="pt", padding=True).input_ids 
                input_ids = input_ids.cuda()
                stsb_ids = model.generate(input_ids, max_new_tokens=1000)
                for decode_i in range(len(stsb_ids)):
                    stsb = tokenizer.decode(stsb_ids[decode_i],skip_special_tokens=True)
                    if answer1_list[decode_i] == answer1_1:
                        answer1_scores.append(5.0)
                    else:
                        if stsb == 'entailment':
                            stsb = '5.0'
                        try: 
                            score = float(stsb)
                        except:
                            score = 0
                        answer1_scores.append(score)

                answer1_scores_list.append(answer1_scores)
            
        

            new_samples.append(sample + [str(answer1_scores_list)])
        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer1_score']
        dataloader.write_dataset(file, head, new_samples)    
            


if __name__ == '__main__':
    args = parse_option()

    main(args)