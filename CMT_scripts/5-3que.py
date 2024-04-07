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
    '''1. The source domain is the concept area from which the metaphor is drawn.
2. The concepts in source domain are typically concrete.
3. The source domain is the domain of experience or concepts that are more concrete, tangible, and familiar to us. It serves as the basis for understanding or talking about a less concrete or abstract concept, which is referred to as the target domain. The source domain provides the metaphorical elements or framework through which we comprehend the target domain.
4. The source domain is a conceptual domain. Conceptual domains are sets of value meanings (presented using a list of concepts or a description of the members of the set) and are used to describe the set of concepts that can be represented within a data element. 
5. For example, in the metaphor "I've invested a lot of time in her," the source domain is "money" and we draw upon our understanding of money to make sense of the concept of time in terms of value, efficiency, and spending.
''',
'''1. Target domain is used for the concept area to which the metaphor is applied.
2. The concepts in the target domain are typically vague and abstract.
3. The target domain is a conceptual domain. Conceptual domains are sets of value meanings (presented using a list of concepts or a description of the members of the set) and are used to describe the set of concepts that can be represented within a data element. 
4. For example, in the metaphor "I've invested a lot of time in her," the target domain is "time", which is being conceptualized in terms of the source domain of money.
'''
]
pos_to_word = {
    'verb':'verb',
}
system_prompt = '''You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09
Current date: 2023-10-15'''



def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='results/CMT-1que-2que-add-kdg-1.0-vrto0.8-sim2.0-5-3/gpt-3.5-turbo-0613/', type=str)
    parser.add_argument('--baseline', default='CMT-3que', type=str, help='')
    parser.add_argument('--vote_number', default=5, type=int)
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results/', type=str)
    parser.add_argument('--vote_ratio', default=0.8, type=float) # 1 时相当于 所有的词义对都需要满足相似
    parser.add_argument('--sentence_similar_ratio', default=2.0, type=float)
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
        new_samples = []
        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text, soure_domain_list_text, target_domain_list_text, soure_domain_score_text, target_domain_score_text, first_final_answer, second_final_answer = sample
            sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')
            answers = eval(answer_text)
            answer_adds = eval(answer_add_text)

            if first_final_answer == '':
                first_final_answer = random.sample(answers[0], 1)
            if second_final_answer == '':
                second_final_answer = random.sample(answers[0], 1)

            prompt_contrast = '''Based on the above information, decide whether the source domain is different from the target domain?'''
            prompt_sentence_context = 'Here is a sentence "%s". Now we know the source domain implied by the %s "%s" and the target domain that the word "%s" tries to describe in this sentence.\n\n' %(sentence_add_punch, pos_to_word[pos_tag.lower()], target_word, target_word)

            dialogs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_sentence_context + first_final_answer + '\n\n' + second_final_answer + '\n\n' + prompt_contrast}
                ]
        
            print(dialogs)

            response = openai.ChatCompletion.create(
            model=_model,
            messages=dialogs,
                # max_tokens=10,
                n=args.vote_number,
                temperature=args.temp,
            )
            answers_res = []
            for answer_add_i in range(args.vote_number):
                answers_res.append(response['choices'][answer_add_i]['message']['content'])
            answers.append(answers_res)


            new_samples.append(sample[:7]+ [sample[8] + str(dialogs)] + [str(answers)] + [str(answer_adds)] + sample[10:])
            head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer2_extract', 'answer1_sim_score', 'answer2_sim_score', 'answer1_final','answer2_final']
            dataloader.write_dataset(file, head, new_samples)




if __name__ == '__main__':
    args = parse_option()

    main(args)