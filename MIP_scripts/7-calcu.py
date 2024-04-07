#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: no knowledge prompt for question 3, get the final answer for question 3 with majority vote
'''

import argparse
import os
import csv
import random
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    accuracy_score
)
class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.baseline, args._model, args.dataset)
    
    def get_dataset(self, file):
        file_dir = os.path.join(self.data_root_path, file)
        examples = []
        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)  # skip the headline
            for i, line in enumerate(lines):
                # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                # example sentence may be empty, caution for processing
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
dataset_to_fileList = {
    'MOH-X':['MOH-X.csv'],
    'TroFi':['TroFi.csv'],
}

def parse_option():
    parser = argparse.ArgumentParser(description='Train on MOH-X dataset, do cross validation')
    parser.add_argument('--data_dir', default='results/MIP-3que-add-kdg-1.0-vrto0.6-sim5.0-7-3/gpt-3.5-turbo-0613', type=str)
    parser.add_argument('--_model', default='gpt-3.5-turbo-0613', type=str, help='gpt-3.5-turbo-0613')
    parser.add_argument('--baseline', default='MIP-calcu', type=str)
    parser.add_argument('--vote_number', default=7, type=int)
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results-calcu/', type=str)

    args, unparsed = parser.parse_known_args()
    return args

def main(args):
    dataloader = Data_Processor(args)
    for file in dataset_to_fileList[args.dataset]:
        samples = dataloader.get_dataset(file)
        ground_truth_list = []
        predict_list = []
        for sample_id in range(len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text, soure_domain_list_text, target_domain_list_text, soure_domain_score_text, target_domain_score_text, first_final_answer, second_final_answer, third_final_answer = sample
            ground_truth_list.append(int(label))
            if third_final_answer == '':
                final_answer3 = random.sample(['exist', 'non-exist'], 1)
            else:
                final_answer3 = third_final_answer
            # compare the graph
            if first_final_answer != '' and second_final_answer != '' and final_answer3 == 'exist':
                predict_list.append(1)
            else:
                predict_list.append(0)
        print(classification_report(np.array(ground_truth_list), np.array(predict_list), digits=4))
        print(precision_score(np.array(ground_truth_list), np.array(predict_list), average='macro'), recall_score(np.array(ground_truth_list), np.array(predict_list), average='macro'), f1_score(np.array(ground_truth_list), np.array(predict_list), average='macro'), accuracy_score(np.array(ground_truth_list), np.array(predict_list)))





if __name__ == '__main__':
    args = parse_option()

    main(args)