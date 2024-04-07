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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='results/CMT-3que-1.0-vrto0.8-sim2.0-5-3/gpt-3.5-turbo-0613', type=str)
    parser.add_argument('--_model', default='gpt-3.5-turbo-0613', type=str, help='gpt-3.5-turbo-0613')
    parser.add_argument('--baseline', default='CMT-calcu', type=str)
    parser.add_argument('--vote_number', default=5, type=int)
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
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text, soure_domain_list_text, target_domain_list_text, soure_domain_score_text, target_domain_score_text, first_final_answer, second_final_answer = sample
            ground_truth_list.append(int(label))
            answers = eval(answer_text)
            contrast_answers = answers[2]
            vote_answer3_exist = 0
            vote_answer3_not_exist = 0
            final_answer3 = ''
            for contrast_answer in contrast_answers:
                if 'Yes,' in contrast_answer or 'Yes' == contrast_answer or 'Yes.' in contrast_answer or 'is different from' in contrast_answer or 'are different' in contrast_answer or 'is indeed different from' in contrast_answer or 'is likely different from' in contrast_answer or 'be different from' in contrast_answer or 'be different from' in contrast_answer  or 'are not significantly different' in contrast_answer or 'are indeed different' in contrast_answer or 'Therefore, in this context, the source domain is the extreme heat, and the target domain is the countryside being scorched.' in contrast_answer or 'are distinct' in contrast_answer or 'are likely different' in contrast_answer or 'appear to be different' in contrast_answer or 'is not inherently different from' in contrast_answer or ', while the target domain refers to' in contrast_answer or 'differs from' in contrast_answer or ", while the target domain represents the conceptual domain associated with personal change or profound transformation." in contrast_answer or 'is distinct from' in contrast_answer or 'can be considered different' in contrast_answer or 'which may differ from' in contrast_answer or 'appears to differ from' in contrast_answer or 'may indeed be different' in contrast_answer or 'do not align' in contrast_answer or 'is most likely different from' in contrast_answer or 'the source domain and the target domain differ in this context' in contrast_answer or 'is conceptually different from' in contrast_answer or 'is indeed different' in contrast_answer or 'can be different' in contrast_answer or 'do differ' in contrast_answer or 'there is a distinct' in contrast_answer or 'can indeed be different' in contrast_answer or 'seem to be different' in contrast_answer or 'is somewhat different from' in contrast_answer or 'might be different' in contrast_answer or 'is separate from' in contrast_answer or 'source domain and the target domain differ from each other' in contrast_answer or 'may be different' in contrast_answer or 'the source domain (physical or spatial domain) and the target domain (altitude) refer to different aspects' in contrast_answer or 'is not the same as' in contrast_answer or 'does differ from' in contrast_answer or 'make the source domain different from the target domain' in contrast_answer or 'are not the same' in contrast_answer or 'would differ' in contrast_answer or 'could potentially be different' in contrast_answer or 'is possible for the source domain and target domain to be different' in contrast_answer  or 'are not entirely the same' in contrast_answer or 'the source and target domains differ' in contrast_answer or 'is a difference between' in contrast_answer or 'may differ' in contrast_answer or 'is generally different from' in contrast_answer or 'is not inherently the same as' in contrast_answer or 'can be considered distinct from' in contrast_answer or ' are conceptually different' in contrast_answer or 'the source domain and the target domain differ' in contrast_answer: 
                    vote_answer3_exist += 1
                elif 'No,' in contrast_answer or 'are the same' in contrast_answer or 'is the same as' in contrast_answer or 'is not different from' in contrast_answer or 'aligns with' in contrast_answer or 'are closely related' in contrast_answer or 'are related but not necessarily different' in contrast_answer or 'not necessarily different from' in contrast_answer or 'is likely the same as' in contrast_answer or 'would be the same' in contrast_answer or 'is not necessarily different from' in contrast_answer or 'is closely related to' in contrast_answer or 'are related' in contrast_answer or 'closely related' in contrast_answer or 'appear to be the same' in contrast_answer or 'are likely the same' in contrast_answer or 'appears to align with' in contrast_answer or 'is similar to' in contrast_answer or 'the target domain align' in contrast_answer or 'matches the target domain' in contrast_answer or 'is not fundamentally different from' in contrast_answer or 'are aligned' in contrast_answer or 'does not differ from' in contrast_answer or 'are indeed the same' in contrast_answer or 'is not significantly different from ' in contrast_answer or 'are actually the same' in contrast_answer or 'are essentially the same' in contrast_answer or 'are not inherently different' in contrast_answer or 'seems to align with' in contrast_answer or 'seem to be the same' in contrast_answer or 'are not necessarily distinct' in contrast_answer or 'appears to be the same as' in contrast_answer or 'are not necessarily different' in contrast_answer or 'the source domain and the target domain refer to the same' in contrast_answer or 'appear to be aligned' in contrast_answer or 'appears to align closely with' in contrast_answer or 'are not distinct' in contrast_answer  or 'seem to pertain' in contrast_answer or 'Both domains pertain to' in contrast_answer or 'there is no clear distinction or difference between the source domain and the target domain' in contrast_answer or 'This is not different' in contrast_answer or 'the source and target domains align' in contrast_answer or 'are not explicitly different' in contrast_answer or 'are one and the same as' in contrast_answer or '''there isn't a distinct''' in contrast_answer or 'is no clear distinction between' in contrast_answer or 'it describes is not necessarily different' in contrast_answer or 'is no apparent differentiation between' in contrast_answer or 'is not distinct from' in contrast_answer or 'is a subset or specific instance within the broader source domain' in contrast_answer or 'the source domain and target domain align' in contrast_answer or 'does not appear to be significantly different' in contrast_answer or 'the source domain and the target domain are similar' in contrast_answer or 'target domain is a specific aspect or subject within the broader source domain' in contrast_answer or 'there is no distinction between the source and target domains' in contrast_answer or 'may not be different' in contrast_answer or 'the target domain is directly related to the source domain' in contrast_answer or 'does not necessarily indicate' in contrast_answer or 'are same' in contrast_answer or 'can be considered the same' in contrast_answer or 'the source domain and the implied target domain may overlap' in contrast_answer or "doesn't seem to be a clear distinction" in contrast_answer or 'is no apparent metaphorical relation' in contrast_answer or 'are not fundamentally different' in contrast_answer or 'is no clear indication' in contrast_answer or 'overlaps with the target domain' in contrast_answer or 'is not entirely different from' in contrast_answer or 'are the relevant aspects' in contrast_answer or 'does not align with' in contrast_answer or 'do not necessarily differ' in contrast_answer or 'appear to be similar rather than different' in contrast_answer or 'are closely connected' in contrast_answer or 'not different from' in contrast_answer or 'can be considered similar to' in contrast_answer or 'is no distinction between' in contrast_answer or 'may overlap' in contrast_answer or 'is closely aligned with' in contrast_answer or 'there is overlap between' in contrast_answer or 'is directly related to' in contrast_answer or 'are effectively the same domain' in contrast_answer or 'the same as the target domain' in contrast_answer or 'are not inherently distinct or different' in contrast_answer or 's likely that the source domain and the target domain share a similarity' in contrast_answer or 'the source domain and the target domain share a commonality' in contrast_answer or 'the source and target domains overlap' in contrast_answer or 'the source domain and the possible target domain in this sentence could be related' in contrast_answer or 'are likely related and not different' in contrast_answer or 'align as both' in contrast_answer or 'are not clearly distinct or different' in contrast_answer or 'share some similarities' in contrast_answer: 
                    vote_answer3_not_exist+=1

                elif "it's difficult to provide a definitive answer" in contrast_answer or 'without further context or information' in contrast_answer or 'without more context' in contrast_answer or 'provide more context' in contrast_answer or 'without additional context' in contrast_answer.lower() or 'it is not possible to determine' in contrast_answer or 'without further context' in contrast_answer.lower():
                    continue
                elif "I will do my best to provide accurate and helpful responses while adhering to these guidelines." in contrast_answer or "I will do my best to provide helpful and informative responses while adhering to your guidelines." in contrast_answer or "I'll do my best to provide accurate" in contrast_answer or 'I will do my best to provide accurate' in contrast_answer or 'not necessarily contrast' in contrast_answer or 'Without additional information or clarification' in contrast_answer or 'are not different' in contrast_answer or 'would likely be the same as' in contrast_answer or 'are closely aligned' in contrast_answer:
                    continue
                else:
                    print("need to add new rules", contrast_answer)
                    
                    continue
            
            if vote_answer3_exist == vote_answer3_not_exist:
                final_answer3 = random.sample(['exist', 'non-exist'], 1)
            elif vote_answer3_not_exist > vote_answer3_exist:
                final_answer3 = 'non-exist'
            elif vote_answer3_not_exist < vote_answer3_exist:
                final_answer3 = 'exist'
            
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