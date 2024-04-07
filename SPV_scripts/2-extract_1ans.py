#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import csv

dataset_to_fileList = {
    'MOH-X':['MOH-X.csv'],
    'TroFi':['TroFi.csv'],
}


def parse_option():
    parser = argparse.ArgumentParser(description='Train on MOH-X dataset, do cross validation')
    parser.add_argument('--data_dir', default='results/', type=str)
    parser.add_argument('--ckpt_dir', default='gpt-3.5-turbo-0613', type=str, help='gpt-3.5-turbo-0613, text-ada-001')
    parser.add_argument('--baseline', default='SPV-1que-1.0', type=str, help='')
    parser.add_argument('--output_baseline', default='SPV-1que-1.0-extract', type=str, help='')
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results/', type=str)

    args, unparsed = parser.parse_known_args()
    return args


class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.baseline, args.ckpt_dir, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.output_baseline, args.ckpt_dir, args.dataset)
    
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
    print(args.data_dir, args.baseline, args.ckpt_dir, args.dataset)
    for file in dataset_to_fileList[args.dataset]:
        samples = dataloader.get_dataset(file)
        for i, sample in enumerate(samples):
            vote_metaphor = 0
            vote_literal = 0
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text = sample
            answers = eval(answer_text)

            source_extract_list = []
            for source_ori in answers[0]:
                if '1.' in source_ori:
                    source_ori = source_ori[source_ori.index('1.'):]
                print(source_ori)
                source_extract_list.append(source_ori)

            new_samples.append(sample[:10] + [str(source_extract_list)])
        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract',]
        dataloader.write_dataset(file, head, new_samples)    
            

if __name__ == '__main__':
    args = parse_option()
    main(args)
