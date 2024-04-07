#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import csv
import openai

openai.api_key = "your_openai_key"


class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.method, args._model, args.dataset)
    
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
                    examples.append(line)
            return examples
dataset_to_fileList = {
    'MOH-X':['MOH-X.csv'],
    'TroFi':['TroFi.csv'],
}


pos_to_word = {
    'verb':'verb',
}
system_prompt = '''You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09
Current date: 2023-10-15'''

source_domain_prompt = '''The source domain is a conceptual domain containing concepts that are typically concrete, tangible, and familiar to us. What is the source domain implied by the [pos] "[target_word]"?'''

target_domain_prompt = '''The target domain is a conceptual domain containing concepts that are typically vague and abstract. In the sentence "[sentence]", what is the target domain that the word "[target_word]" tries to describe?'''


def parse_option():
    parser = argparse.ArgumentParser(description='Metaphor reasoning')
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--method', default='CMT-1que-2que', type=str, help='')
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--output_path', default='results/', type=str)
    parser.add_argument('--_model', default='gpt-3.5-turbo-0613', type=str, help='''gpt-3.5-turbo-0613''')
    parser.add_argument('--N_o', default=5, type=int)
    args, unparsed = parser.parse_known_args()
    return args



def main(args):


    args.method = args.method + '-' + str(args.temp) 
    _model = args._model
    dataloader = Data_Processor(args)

    for file in dataset_to_fileList[args.dataset]:
        new_samples = []
        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent = sample
            sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"')
            answers = [[], []]
            answers_kdg = [[], []]
            questions = [
                source_domain_prompt.replace('[pos]', pos_to_word[pos_tag.lower()]).replace('[target_word]', target_word),
                target_domain_prompt.replace('[sentence]', sentence_add_punch).replace('[target_word]', target_word),
            ]

            for question_index, question in enumerate(questions):

                dialogs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": questions[question_index]}
                    ]
                response = openai.ChatCompletion.create(
                model=_model,
                messages=dialogs,
                    # max_tokens=10,
                    n=args.N_o,
                    temperature=args.temp,
                )
                for add_i in range(args.N_o):
                    answers[question_index].append(response['choices'][add_i]['message']['content'])
            
            new_samples.append([sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,str(questions),str(answers),str(answers_kdg)])
            head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg']
            dataloader.write_dataset(file, head, new_samples)


if __name__ == '__main__':
    args = parse_option()
    main(args)