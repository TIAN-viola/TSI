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

def third_answer_filter(input_answer_upper):
    input_answer = input_answer_upper.lower()
    if 'is different from' in input_answer or 'goes beyond the basic meaning' in input_answer or  '" differs from' in input_answer or 'is a specific application of its basic meaning' in input_answer or 'rather than its basic literal meaning' in input_answer or 'is a nuanced extension of its basic meaning' in input_answer or 'is essentially an extension' in input_answer or 'is slightly different from the basic meaning' in input_answer or 'is slightly different from its basic meaning' in input_answer or 'the contextual meaning differs from the basic meaning' in input_answer or 'has a different contextual meaning compared to its basic meaning' in input_answer or 'are different' in input_answer or 'does not align with' in input_answer or 'does differ from' in input_answer or 'is indeed different from' in input_answer or 'differs slightly from' in input_answer or 'differs from' in input_answer or 'does have a slightly different meaning' in input_answer or 'does have a slightly different' in input_answer or 'appears to be different from' in input_answer or ' does appear to be different from' in input_answer or 'does have a slightly different meaning' in input_answer or 'its contextual meaning in this case is different' in input_answer or 'does indeed differ from' in input_answer or 'the contextual meaning in this sentence is different' in input_answer or 'does not appear to align with' in input_answer or 'be a slight nuance in the specific connotation' in input_answer or 'does differ slightly from' in input_answer or 'may differ slightly from' in input_answer or 'is not entirely different from' in input_answer or 'is a figurative extension of' in input_answer or 'extends beyond its basic meaning' in input_answer or 'has a different meaning from' in input_answer or 'has a slightly different meaning from' in input_answer or 'is indeed different' in input_answer or 'seems to be significantly different from' in input_answer or 'appears to differ from' in input_answer or 'slightly deviates from' in input_answer or 'may differ from' in input_answer or 'seems to be different from' in input_answer or 'has a slightly different meaning' in input_answer or 'has a different connotation' in input_answer or ', rather than the movement of substances like liquids or gases' in input_answer or 'has a different' in input_answer or 'does seem to differ from' in input_answer or 'might indeed be different from' in input_answer or 'does deviate from' in input_answer or 'may be different from' in input_answer or 'carries a slightly different' in input_answer or 'does deviate slightly from' in input_answer or 'could be different from' in input_answer or 'does have a different' in input_answer or ', which extends beyond' in input_answer or 'is specific and different from' in input_answer or 'conveys a nuanced meaning beyond' in input_answer or 'seems different from' in input_answer or 'does not have a different' in input_answer or 'is likely different from' in input_answer or 'diverges from' in input_answer or 'can be considered different from' in input_answer or 'and different from its basic meaning' in input_answer or 'differs slightly' in input_answer or 'goes beyond its basic meaning' in input_answer or 'could potentially be different from' in input_answer or 'is used in a slightly different sense than' in input_answer or 'deviates from' in input_answer or 'beyond its basic meaning' in input_answer or 'deviates slightly from' in input_answer or 'is significantly different from' in input_answer or 'is somewhat different from' in input_answer or 'is a significant difference between' in input_answer or 'might be different from' in input_answer or 'seems to deviate from' in input_answer or 'takes on a slightly different meaning' in input_answer:
        return 1
    
    # elif 'No.' in input_answer or 'No,' in input_answer or 
    elif '" align' in input_answer or 'does contrast with' in input_answer or 'align rather than contrast' in input_answer or 'align quite closely' in input_answer or 'does not contrast with' in input_answer or 'not necessarily contrast with' in input_answer or 'and its contextual meaning align' in input_answer or 'aligns with' in input_answer or 'not contrast' in input_answer or 'are aligned' in input_answer or 'not contradictory' in input_answer or 'would not contrast' in input_answer or 'not in contrast' in input_answer or 'both the basic and contextual meanings align' in input_answer or 'does align' in input_answer or 'is essentially the same as' in input_answer or ' is the same as' in input_answer or 'is consistent with' in input_answer or 'retains the same basic meaning as its contextual meaning' in input_answer or 'retains its basic meaning' in input_answer or 'is a specific application or usage of' in input_answer or 'is a specific application or extension' in input_answer or 'a specific application or extension' in input_answer or 'is a specific application of' in input_answer or 'is not different from' in input_answer or 'is closely related to' in input_answer or 'is a specific manifestation of' in input_answer  or 'aligns closely with' in input_answer or 'is aligned with' in input_answer or 'does not appear to be different from' in input_answer or 'does not seem to be different from' in input_answer or 'does not differ significantly from' in input_answer or 'does not seem to differ from' in input_answer or 'suggests a specific application of' in input_answer or 'seems to align with' in input_answer or 'is not substantially different from' in input_answer or 'appears to align with' in input_answer or 'does not seem to differ significantly from' in input_answer or 'does not have a different meaning from' in input_answer or 'the basic meaning and the contextual meaning align in this case' in input_answer or 'does not appear to differ' in input_answer or 'does not appear to differ from' in input_answer or 'does not differ from' in input_answer or 'does not have a different contextual meaning' in input_answer or 'remains consistent with' in input_answer or 'would align with' in input_answer or 'is similar to' in input_answer or 'does not deviate significantly from' in input_answer or 'seems consistent with' in input_answer or 'does not appear to deviate from' in input_answer or 'does not significantly differ from' in input_answer or 'is not significantly different from' in input_answer or 'is not fundamentally different from' in input_answer or 'does not deviate from' in input_answer or 'matches the basic meaning' in input_answer or '''doesn't seem to be a significant difference''' in input_answer or 'is an extension of ' in input_answer or 'is no significant difference' in input_answer or 'does not appear to be significantly different' in input_answer or 'is synonymous with' in input_answer or 'does not alter' in input_answer or 'while the basic meaning remains the same' in input_answer or 'does not suggest a different contextual meaning' in input_answer or 'does not suggest a different' in input_answer or 'does not appear to deviate significantly from' in input_answer or 'does not seem to deviate significantly from' in input_answer or 'coincides with its contextual meaning' in input_answer or 'does not seem to deviate from' in input_answer or 'does not seem to be significantly different' in input_answer or 'matches its basic meaning' in input_answer or 'is no distinction' in input_answer or 'does not significantly deviate from' in input_answer or 'is no difference between' in input_answer or 'is essentially consistent with' in input_answer or 'does not differ substantially from' in input_answer or 'is equivalent to' in input_answer or 'retains the same basic meaning' in input_answer or 'maintains its basic meaning' in input_answer or 'is generally consistent with' in input_answer or 'the contextual meaning can be considered a narrower or more specific interpretation of the basic meaning' in input_answer or 'remains the same as' in input_answer or 'does not have a significantly different' in input_answer or 'has the same meaning as' in input_answer or 'remains the same' in input_answer or 'is in line with' in input_answer or 'can be seen as a subset or variation' in input_answer or 'is a specific case within the broader basic meaning' in input_answer or 'is a specific instance of the basic meaning' in input_answer or 'is closely aligned with' in input_answer or 'rather than being fundamentally different' in input_answer or 'are the same' in input_answer or 'is used in its basic and literal meaning' in input_answer or 'does not have a contextual meaning different from its basic meaning' in input_answer or 'does not carry a contextual meaning different from its basic meaning' in input_answer or 'is an extension or application of its basic meaning' in input_answer or 'the word is being used in its basic sense' in input_answer or 'retains a similar meaning' in input_answer or 'remains within the broader basic meaning' in input_answer or 'is largely consistent with' in input_answer or 'carries the same meaning as' in input_answer or 'is not inherently different from' in input_answer or 'is used in its basic meaning' in input_answer or 'corresponds with its basic meaning' in input_answer or 'does not appear to diverge significantly from' in input_answer or 'are not different' in input_answer or 'does appear to differ from' in input_answer or 'has the same contextual meaning' in input_answer or "doesn't appear to be a significant difference" in input_answer or '''doesn't seem to be a difference''' in input_answer: 
        return 0
    # 这条要放到最后，因为有时候模型给出了一个答案，但也说了depend on context
    elif "it's difficult to provide a definitive answer" in input_answer or 'without further context or information' in input_answer or 'without more context' in input_answer or 'provide more context' in input_answer or 'is unclear or different from' in input_answer or 'it is difficult to determine' in input_answer:
        return -1
    elif "I will do my best to provide accurate and helpful responses while adhering to these guidelines." in input_answer or "I will do my best to provide helpful and informative responses while adhering to your guidelines." in input_answer or "I'll do my best to provide accurate" in input_answer or 'I will do my best to provide accurate' in input_answer or 'not necessarily contrast' in input_answer or 'Without additional information or clarification' in input_answer:
        return -1
    else:   
        print(input_answer + '\n=================')
        return -1

def exist_refuse_to_answer(answers):
    refuse_to_answer_list = ['depends on the context', 'it is not possible to determine', 'without additional context', 'vary depending on the context', 'would depend on the specific context', 'without further context', 'additional context or information', 'additional context', 'it is difficult to determine', 'additional information', 'depends on the specific context', 'cannot be determined']
    final_answer = ''
    for answer in answers:
        flag_refuse_to_answer = False
        for refuse_to_answer_phrase in refuse_to_answer_list:
            if refuse_to_answer_phrase in answer.lower():
                flag_refuse_to_answer = True
                break
        if flag_refuse_to_answer == False:
            final_answer = answer
            break
    return final_answer

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
pos_to_word = {
    'verb':'verb',
}
system_prompt = '''You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09
Current date: 2023-10-15'''

third_question_prompt = '''Based on above information, decide whether the contextual meaning of the word "[target_word]" is different from its basic meaning?'''

first_two_que_context_prompt = '''Here is a sentence "[sentence]". Now we know the basic meaning of the [pos] "[target_word]" and the contextual meaning of the word "[target_word]".\n\n'''
def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='results/MIP-3que-1.0-kvrto0.6-sim5.0-7-3/gpt-3.5-turbo-0613/', type=str)
    parser.add_argument('--baseline', default='MIP-3que-add-kdg', type=str, help='')
    parser.add_argument('--vote_number', default=7, type=int)
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results/', type=str)
    parser.add_argument('--vote_ratio', default=0.6, type=float) # 1 时相当于 所有的词义对都需要满足相似
    parser.add_argument('--sentence_similar_ratio', default=5.0, type=float)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--add_kdg_number', default=3, type=int)
    parser.add_argument('--_model', default='gpt-3.5-turbo-0613', type=str, help='''gpt-3.5-turbo-0613''')
    args, unparsed = parser.parse_known_args()
    return args



def main(args):

    _model = args.data_dir.split('/')[-2]
    args._model = _model
    args.baseline = args.baseline + '-' + str(args.temp) +  '-vrto' + str(args.vote_ratio)+ '-sim' + str(args.sentence_similar_ratio) + '-' + str(args.vote_number) + '-' + str(args.add_kdg_number)

    dataloader = Data_Processor(args)
    vote_baseline_number = round(args.vote_number * args.vote_ratio)
    for file in dataset_to_fileList[args.dataset]:
        dialogs = []
        new_samples = []
        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text, soure_domain_list_text, target_domain_list_text, soure_domain_score_text, target_domain_score_text, first_answer_final, second_answer_final = sample
            sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')

            if first_answer_final == '':
                first_answer_final = random.sample(answers[0], 1)
            if second_answer_final == '':
                second_answer_final = random.sample(answers[1], 1)
            
            prompt_contrast = third_question_prompt.replace('[target_word]', target_word)
            prompt_sentence_context = first_two_que_context_prompt.replace('[sentence]', sentence_add_punch).replace('[target_word]', target_word).replace('[pos]', pos_to_word[pos_tag.lower()]) 
            prompt_third = prompt_sentence_context + first_answer_final + '\n\n' + second_answer_final + '\n\n' + prompt_contrast

            # third_answer
            third_final_answer = ''
            answers = eval(answer_text)
            answer_adds = eval(answer_add_text)
            third_answers = answers[2]
            different_count = 0
            non_different_count = 0
            for third_answer in third_answers:
                third_answer_index = third_answer_filter(third_answer)
                if third_answer_index == 1:
                    different_count += 1
                elif third_answer_index == 0:
                    non_different_count += 1
            if different_count >= vote_baseline_number:
                third_final_answer = 'exist'
            elif non_different_count >= vote_baseline_number:
                third_final_answer = 'non-exist'
            else:
                # need knowledge
                prompt_third_add_knowledge = knowledge_list[2] + '\n' + 'Taking the knowledge provided above into account, please answer the following question:\n' + prompt_third.replace('[pos]', pos_to_word[pos_tag.lower()]).replace('[target_word]', target_word)
                if args._model in ['gpt-3.5-turbo-0613']:
                    dialogs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_third_add_knowledge}
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
                    answer_adds.append(answers_res) 
                    

                    third_final_answer = exist_refuse_to_answer(answers_res)
                else:
                    print('no such model')
            new_samples.append(sample[:7]+ [str(dialogs)] + [str(answers)] + [str(answer_adds)] + sample[10:] + [third_final_answer])
            head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer2_extract', 'answer1_sim_score', 'answer2_sim_score', 'answer1_final','answer2_final','answer3_final']
            dataloader.write_dataset(file, head, new_samples)


if __name__ == '__main__':
    args = parse_option()

    main(args)