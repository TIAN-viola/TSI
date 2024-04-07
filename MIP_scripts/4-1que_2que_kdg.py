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
    '''The basic meaning can be found in general users' dictionaries. The basic meaning is not necessarily the most frequent meaning of the lexical unit. The basic meaning tends to be
- More concrete; what they evoke is easier to imagine, see, hear, feel, smell, and taste.
- Related to bodily action. 
- More precise (as opposed to vague) 
- Historically older.
''',
'''The contextual meaning means the meaning of the lexical unit in context, that is, how the lexical unit applies to an entity, relation, or attribute in the situation evoked by the text. Take into account what comes before and after the lexical unit. The contextual meaning may be conventionalized and will thus be found in a general users' dictionary. It may also be novel or specialized and will thus not be found in a general users' dictionary.
''',
'''If the basic meaning of a word contrasts with its contextual meaning, there is a difference as well as comparison between the contextual and a more basic meaning.
'''
]

source_domain_prompt = '''What is the basic meaning of the [pos] "[target_word]"?'''

target_domain_prompt = '''In the sentence "[sentence]", what is the contextual meaning of the word "[target_word]"?'''

pos_to_word = {
    'verb':'verb',
}
system_prompt = '''You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09
Current date: 2023-10-15'''

def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='results-similar-score/MIP-1que-2que-1.0-extract/gpt-3.5-turbo-0613/', type=str)
    parser.add_argument('--baseline', default='MIP-1que-2que-add-kdg', type=str, help='')
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results/', type=str)
    parser.add_argument('--vote_number', default=7, type=int)
    parser.add_argument('--vote_ratio', default=0.6, type=float) 
    parser.add_argument('--sentence_similar_ratio', default=5.0, type=float)
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
        count_target_need_knowledge = 0
        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text, soure_domain_list_text, target_domain_list_text, soure_domain_score_text, target_domain_score_text = sample
            sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')
            answers = eval(answer_text)
            answer_adds = eval(answer_add_text)
            soure_domain_score_list = np.array(eval(soure_domain_score_text))[0:args.vote_number, 0:args.vote_number]
            soure_domain_score_sort = np.argsort(soure_domain_score_list, axis=None) # 升序
            target_domain_score_list = np.array(eval(target_domain_score_text))[0:args.vote_number, 0:args.vote_number]
            target_domain_score_sort = np.argsort(target_domain_score_list, axis=None) # 升序
            flag_basic_consis = False
            flag_contextual_consis = False
            first_answer_final = ''
            second_answer_final = ''
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

            refuse_to_answer_target_index_list = []
            for refuse_index in range(args.vote_number):
                for refuse_to_answer_phrase in refuse_to_answer_list:
                    if refuse_to_answer_phrase in answers[0][refuse_index].lower():
                        refuse_to_answer_target_index_list.append(refuse_index)
                        break
            target_domain_consistent_element_list = []
            for delete_number in range(args.vote_number - vote_baseline_number + 1):
                if delete_number == 0: # 没有需要删除的行或列
                    if sum(sum(target_domain_score_list >= args.sentence_similar_ratio)) == args.vote_number * args.vote_number:
                        target_domain_consistent_element_list = [i for i in range(args.vote_number)]
                        break
                else:
                    flag = False
                    combins = [c for c in  combinations(range(args.vote_number), delete_number)]
                    if not flag:
                        for combin in combins:
                            if not flag:
                                _target_domain_score_list = copy.deepcopy(target_domain_score_list)
                                target_domain_consistent_element_list_candidate = [i for i in range(args.vote_number)]
                                for i_delete_index, delete_index in enumerate(combin): # (0, 1)
                                    _target_domain_score_list = np.delete(_target_domain_score_list, delete_index-i_delete_index, 0)
                                    _target_domain_score_list = np.delete(_target_domain_score_list, delete_index-i_delete_index, 1)
                                    target_domain_consistent_element_list_candidate.remove(delete_index)
                                if sum(sum(_target_domain_score_list >= args.sentence_similar_ratio)) == (args.vote_number - delete_number) * (args.vote_number - delete_number):
                                    remain_target_domain_consistent_element_list = list(set(target_domain_consistent_element_list_candidate) - set(refuse_to_answer_target_index_list))
                                    if len(remain_target_domain_consistent_element_list) >= vote_baseline_number:
                                        flag = True
                                        target_domain_consistent_element_list = remain_target_domain_consistent_element_list
                                        break

            print(target_domain_consistent_element_list)
            if len(target_domain_consistent_element_list) == 0:
                count_target_need_knowledge+=1
            available_target_domain_list = []
            if target_domain_consistent_element_list!=[]:
                for available_target_domain_index in target_domain_consistent_element_list:
                    available_target_domain_list.append(answers[1][available_target_domain_index])
                flag_contextual_consis = True    
                second_answer_final = random.sample(available_target_domain_list, 1)

            if not flag_basic_consis:
                prompt_basic = knowledge_list[0] + '\n' + 'Taking the knowledge provided above into account, please answer the following question:\n' + source_domain_prompt.replace('[pos]', pos_to_word[pos_tag.lower()]).replace('[target_word]', target_word)
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


            if not flag_contextual_consis:
                prompt_contextual = knowledge_list[1] + '\n' + 'Taking the knowledge provided above into account, please answer the following question:\n' + target_domain_prompt.replace('[sentence]', sentence_add_punch).replace('[target_word]', target_word)
                if args._model in ['gpt-3.5-turbo-0613']:
                    dialogs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_contextual}
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
                    answer_adds[1] = answers_res

                    for answer_res in answers_res:
                        flag_refuse_answer_res = False
                        for refuse_to_answer_phrase in refuse_to_answer_list:
                            if refuse_to_answer_phrase in answer_res.lower():
                                flag_refuse_answer_res = True
                                break
                        if flag_refuse_answer_res == False:
                            second_answer_final = answer_res
                            break

                else:
                    print('no such model')

            new_samples.append(sample[:7]+ [str(dialogs)] + [str(answers)] + [str(answer_adds)] + sample[10:] + [first_answer_final, second_answer_final])
            head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer2_extract', 'answer1_sim_score', 'answer2_sim_score', 'answer1_final','answer2_final']
            dataloader.write_dataset(file, head, new_samples)
        print("count_source_need_knowledge:", count_source_need_knowledge)
        print("count_target_need_knowledge", count_target_need_knowledge)



if __name__ == '__main__':
    args = parse_option()

    main(args)