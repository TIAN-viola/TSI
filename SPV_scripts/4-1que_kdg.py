#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import copy
import csv
import numpy as np
from itertools import combinations
import random
import openai

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

first_prompt = '''Please provide three examples of the most frequent usage of the [pos] "[target_word]".'''

def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='results-similar-score/SPV-1que-1.0-extract/gpt-3.5-turbo-0613/', type=str)
    parser.add_argument('--baseline', default='SPV-1que-add-kdg', type=str, help='')
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results/', type=str)
    parser.add_argument('--vote_number', default=5, type=int)
    parser.add_argument('--vote_ratio', default=0.8, type=float) 
    parser.add_argument('--sentence_similar_ratio', default=1.5, type=float)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--add_kdg_number', default=3, type=int)
    args, unparsed = parser.parse_known_args()
    return args



def main(args):

    _model = args.data_dir.split('/')[-2]
    args._model = _model
    args.baseline = args.baseline + '-' + str(args.temp) +  '-vrto' + str(args.vote_ratio)+ '-sim' + str(args.sentence_similar_ratio) + '-' + str(args.vote_number) + '-' + str(args.add_kdg_number)
    dataloader = Data_Processor(args)

    for file in dataset_to_fileList[args.dataset]:
        dialogs=[]
        new_samples = []
        count_source_need_knowledge = 0

        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text, soure_domain_list_text_ori, soure_domain_score_text = sample
            sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')
            answers_ori = eval(answer_text)
            first_answer_final = ''
            answer_adds = eval(answer_add_text)
            n_answer = len(answers_ori[0])
            index = np.random.choice(np.arange(n_answer), size=args.vote_number, replace=False)
            answers = [np.array(answers_ori[0])[index].tolist(), []]
            soure_domain_score_list = np.array(eval(soure_domain_score_text))[np.array([index.tolist()]).T, [index.tolist()]]
            soure_domain_list_text = np.array(eval(soure_domain_list_text_ori))[index].tolist()
            flag_basic_consis = False
        
            vote_baseline_number = round(args.vote_number * args.vote_ratio)
            refuse_to_answer_list = ['depends on the context', 'it is not possible to determine', 'without additional context', 'vary depending on the context', 'would depend on the specific context', 'without further context', 'additional context or information', 'additional context', 'it is difficult to determine', 'additional information', 'depends on the specific context', 'cannot be determined']
            refuse_to_answer_source_index_list = []
            for refuse_index in range(args.vote_number):
                for refuse_to_answer_phrase in refuse_to_answer_list:
                    if refuse_to_answer_phrase in answers[0][refuse_index].lower():
                        refuse_to_answer_source_index_list.append(refuse_index)
                        break
            source_domain_consistent_element_list = []
            for delete_number in range(args.vote_number - vote_baseline_number + 1):
                if delete_number == 0: # 没有需要删除的行或列
                    if sum(sum(soure_domain_score_list >= args.sentence_similar_ratio)) == args.vote_number * args.vote_number:
                        source_domain_consistent_element_list = [i for i in range(args.vote_number)]
                        break
                else:
                    flag = False
                    combins = [c for c in  combinations(range(args.vote_number), delete_number)]
                    if not flag:
                        for combin in combins:
                            if not flag:
                                _source_domain_score_list = copy.deepcopy(soure_domain_score_list)
                                source_domain_consistent_element_list_candidate = [i for i in range(args.vote_number)]
                                for i_delete_index, delete_index in enumerate(combin): # (0, 1)
                                    _source_domain_score_list = np.delete(_source_domain_score_list, delete_index-i_delete_index, 0)
                                    _source_domain_score_list = np.delete(_source_domain_score_list, delete_index-i_delete_index, 1)
                                    source_domain_consistent_element_list_candidate.remove(delete_index)
                                if sum(sum(_source_domain_score_list >= args.sentence_similar_ratio)) == (args.vote_number - delete_number) * (args.vote_number - delete_number):
                                    remain_source_domain_consistent_element_list = list(set(source_domain_consistent_element_list_candidate) - set(refuse_to_answer_source_index_list))
                                    if len(remain_source_domain_consistent_element_list) >= vote_baseline_number:
                                        flag = True
                                        source_domain_consistent_element_list = remain_source_domain_consistent_element_list
                                        break

            print(source_domain_consistent_element_list)
            if len(source_domain_consistent_element_list) == 0:
                count_source_need_knowledge+=1
            available_source_domain_list = []
            if source_domain_consistent_element_list!=[]:
                for available_source_domain_index in source_domain_consistent_element_list:
                    available_source_domain_list.append(answers[0][available_source_domain_index])
                flag_basic_consis = True
                first_answer_final = random.sample(available_source_domain_list, 1)


            if not flag_basic_consis:
                prompt_basic = knowledge_list[0] + '\n' + 'Taking the knowledge provided above into account, please answer the following question:\n' + first_prompt.replace('[pos]', pos_to_word[pos_tag.lower()]).replace('[target_word]', target_word)
                if args._model in ['gpt-3.5-turbo-0613']:
                    dialogs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_basic}
                        ]
                    response = openai.ChatCompletion.create(
                    model=_model,
                    messages=dialogs,
                        # max_tokens=10,
                        n=args.add_kdg_number,
                        temperature=args.temp,
                    )
                    answers_res = []
                    for answer_add_i in range(args.add_kdg_number):
                        answers_res.append(response['choices'][answer_add_i]['message']['content'])
                    answer_adds[0] = answers_res  
                    
                    
                    for answer_res in answers_res:
                        flag_refuse_answer_res = False
                        for refuse_to_answer_phrase in refuse_to_answer_list:
                            if refuse_to_answer_phrase in answer_res.lower():
                                flag_refuse_answer_res = True
                                break
                        if flag_refuse_answer_res == False:
                            first_answer_final = answer_res
                            break

                else:
                    print('no such model')


            new_samples.append(sample[:7]+ [str(dialogs)] + [str(answers)] + [str(answer_adds)] + sample[10:] + [first_answer_final])
            head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer1_sim_score', 'answer1_final']
            dataloader.write_dataset(file, head, new_samples)
        print("count_source_need_knowledge:", count_source_need_knowledge)




if __name__ == '__main__':
    args = parse_option()

    main(args)