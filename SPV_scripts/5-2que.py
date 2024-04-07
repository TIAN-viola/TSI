#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import csv
import numpy as np
from itertools import combinations
import copy
import openai
import random
openai.api_key = "your_openai_key"




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

knowledge_list = [
    '''The phrase "frequent usage of a word" refers to the regular and common application of a word.
''',
'''In the theory of selectional preference violation, Wilks suggests that metaphor represents a violation of combinatory norms in the linguistic context and that metaphorical expressions can be detected via such violation.
''',
]
pos_to_word = {
    'verb':'verb',
}
system_prompt = '''You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09
Current date: 2023-10-15'''


second_question_prompt = '''In the sentence "[sentence]", decide whether the context of the word "[target_word]" is different from its contexts in the above sentences.'''

first_prompt_context = '''Now we know examples of the frequent usage of the [pos] "[target_word]".\n\n'''



def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='results/SPV-1que-add-kdg-1.0-vrto0.8-sim1.5-5-3/gpt-3.5-turbo-0613/', type=str)
    parser.add_argument('--baseline', default='SPV-2que', type=str, help='')
    parser.add_argument('--vote_number', default=5, type=int)
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results/', type=str)
    parser.add_argument('--vote_ratio', default=0.8, type=float) # 1 时相当于 所有的词义对都需要满足相似
    parser.add_argument('--sentence_similar_ratio', default=1.5, type=float)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--add_kdg_number', default=3, type=int)
    parser.add_argument('--_model', default='gpt-3.5-turbo-0613', type=str, help='''gpt-3.5-turbo-0613''')
    args, unparsed = parser.parse_known_args()
    return args



def main(args):


    _model = args._model
    args.baseline = args.baseline + '-' + str(args.temp) +  '-vrto' + str(args.vote_ratio)+ '-sim' + str(args.sentence_similar_ratio) + '-' + str(args.vote_number) + '-' + str(args.add_kdg_number)
    dataloader = Data_Processor(args)
    
    for file in dataset_to_fileList[args.dataset]:
        dialogs = []
        count_source_need_knowledge = 0
        count_target_need_knowledge = 0
        new_samples = []
        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text, soure_domain_list_text, soure_domain_score_text, first_answer_final = sample
            sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')
            answers = eval(answer_text)
            answer_adds = eval(answer_add_text)

            if first_answer_final == '':
                first_answer_final = random.sample(answers[0], 1)
            
            prompt_contrast = second_question_prompt.replace('[target_word]', target_word).replace('[sentence]', sentence_add_punch)
            prompt_sentence_context = first_prompt_context.replace('[sentence]', sentence_add_punch).replace('[target_word]', target_word).replace('[pos]', pos_to_word[pos_tag.lower()]) 
            if args._model in ['gpt-3.5-turbo-0613']:

                dialogs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_sentence_context + first_answer_final + '\n\n' + prompt_contrast}
                        ]


                response = openai.ChatCompletion.create(
                model=_model,
                messages=dialogs,
                    # max_tokens=10,
                    n=args.vote_number,
                    temperature=args.temp
                )
                answers_res = []
                for answer_add_i in range(args.vote_number):
                    answers_res.append(response['choices'][answer_add_i]['message']['content'])
                answers.append(answers_res)
                    

            else:
                print('no such model')

            new_samples.append(sample[:7]+ [sample[8] + str(dialogs)] + [str(answers)] + [str(answer_adds)] + sample[10:])
            head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer1_sim_score', 'answer1_final']
            dataloader.write_dataset(file, head, new_samples)
        print("count_source_need_knowledge", count_source_need_knowledge)
        print("count_target_need_knowledge", count_target_need_knowledge)




if __name__ == '__main__':
    args = parse_option()

    main(args)