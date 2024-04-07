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
    parser = argparse.ArgumentParser(description='-')
    parser.add_argument('--data_dir', default='results/', type=str)
    parser.add_argument('--model_dir', default='gpt-3.5-turbo-0613', type=str, help='gpt-3.5-turbo-0613')
    parser.add_argument('--method', default='CMT-1que-2que-1.0', type=str, help='')
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--output_path', default='results/', type=str)
    args, unparsed = parser.parse_known_args()
    return args


class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.method, args.model_dir, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.output_method, args.model_dir, args.dataset)
    
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


def get_target(target_description):
    # extract basic meaning

    if target_description[-1] != '.':
        target_description = target_description + '.'
    if ' tries to describe in the given sentence is ' in target_description:
        basic_index = target_description.index(' tries to describe in the given sentence is ')
        _target = target_description[basic_index+44:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' describe the target domain of ' in target_description:
        basic_index = target_description.index(' describe the target domain of ')
        _target = target_description[basic_index+31:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' describes the target domain of ' in target_description:
        basic_index = target_description.index(' describes the target domain of ')
        _target = target_description[basic_index+32:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' describe the ' in target_description:
        basic_index = target_description.index(' describe the ')
        _target = target_description[basic_index+14:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' could be ' in target_description:
        basic_index = target_description.index(' could be ')
        _target = target_description[basic_index+10:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' would be ' in target_description:
        basic_index = target_description.index(' would be ')
        _target = target_description[basic_index+10:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' would most likely be ' in target_description:
        basic_index = target_description.index(' would most likely be ')
        _target = target_description[basic_index+2:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' can be interpreted as ' in target_description:
        basic_index = target_description.index(' can be interpreted as ')
        _target = target_description[basic_index+23:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index] 
    elif ' tries to describe in the sentence "' in target_description:
        basic_index = target_description.index(' tries to describe in the sentence "')
        _target = target_description[basic_index+36:]

        if ' is ' in _target:
            basic_index = _target.index(' is ')
            final_target = _target[basic_index+4:]
            _period_index = final_target.index('.')
            final_target = final_target[:_period_index]
        elif ' would likely be ' in _target:
            basic_index = _target.index(' would likely be ')
            final_target = _target[basic_index+17:]
            _period_index = final_target.index('.')
            final_target = final_target[:_period_index]
        else:
            final_target = _target
    elif ' to suggest that ' in target_description:
        basic_index = target_description.index(' to suggest that ')
        _target = target_description[basic_index+17:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' describes the ' in target_description:
        basic_index = target_description.index(' describes the ')
        _target = target_description[basic_index+15:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' is describing is ' in target_description:
        basic_index = target_description.index(' is describing is ')
        _target = target_description[basic_index+18:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif 'describes is ' in target_description:
        basic_index = target_description.index('describes is ')
        _target = target_description[basic_index+13:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' to describe is ' in target_description:
        basic_index = target_description.index(' to describe is ')
        _target = target_description[basic_index+16:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' to describe ' in target_description:
        basic_index = target_description.index(' to describe ')
        _target = target_description[basic_index+13:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' could metaphorically describe ' in target_description:
        basic_index = target_description.index(' could metaphorically describe ')
        _target = target_description[basic_index+31:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' may describe ' in target_description:
        basic_index = target_description.index(' may describe ')
        _target = target_description[basic_index+14:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' to mean ' in target_description:
        basic_index = target_description.index(' to mean ')
        _target = target_description[basic_index+9:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' is the ' in target_description:
        basic_index = target_description.index(' is the')
        _target = target_description[basic_index+8:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]    
    elif ' is likely related to ' in target_description:
        basic_index = target_description.index(' is likely related to')
        _target = target_description[basic_index+22:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index] 
    elif ' is likely the ' in target_description:
        basic_index = target_description.index(' is likely the ')
        _target = target_description[basic_index+15:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' is used to describe ' in target_description:
        basic_index = target_description.index(' is used to describe ')
        _target = target_description[basic_index+21:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' is used in the context of ' in target_description:
        basic_index = target_description.index(' is used in the context of ')
        _target = target_description[basic_index+27:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif '" is ' in target_description:
        basic_index = target_description.index('" is ')
        _target = target_description[basic_index+5:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' is that of ' in target_description:
        basic_index = target_description.index(' is that of ')
        _target = target_description[basic_index+12:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' is ' in target_description:
        basic_index = target_description.index(' is ')
        _target = target_description[basic_index+4:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' tries to describe ' in target_description:
        basic_index = target_description.index(' tries to describe ')
        _target = target_description[basic_index+19:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' describes ' in target_description:
        basic_index = target_description.index(' describes ')
        _target = target_description[basic_index+11:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' refers to ' in target_description:
        basic_index = target_description.index(' refers to ')
        _target = target_description[basic_index+11:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' implies ' in target_description:
        basic_index = target_description.index(' implies ')
        _target = target_description[basic_index+9:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' involves ' in target_description:
        basic_index = target_description.index(' involves ')
        _target = target_description[basic_index+10:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' suggests ' in target_description:
        basic_index = target_description.index(' suggests ')
        _target = target_description[basic_index+10:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' means to ' in target_description:
        basic_index = target_description.index(' means to ')
        _target = target_description[basic_index+10:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' would be ' in target_description:
        basic_index = target_description.index(' would be ')
        _target = target_description[basic_index+10:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' would likely be ' in target_description:
        basic_index = target_description.index(' would likely be ')
        _target = target_description[basic_index+17:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' related to ' in target_description:
        basic_index = target_description.index(' related to ')
        _target = target_description[basic_index+11:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' relates to ' in target_description:
        basic_index = target_description.index(' relates to ')
        _target = target_description[basic_index+11:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' appears to be ' in target_description:
        basic_index = target_description.index(' appears to be ')
        _target = target_description[basic_index+15:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' could also relate to ' in target_description:
        basic_index = target_description.index(' could also relate to ')
        _target = target_description[basic_index+22:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' would typically be ' in target_description:
        basic_index = target_description.index(' would typically be ')
        _target = target_description[basic_index+20:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' the target domain of the conceptual domain of ' in target_description:
        basic_index = target_description.index(' the target domain of the conceptual domain of ')
        _target = target_description[basic_index+47:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' could potentially describe ' in target_description:
        basic_index = target_description.index(' could potentially describe ')
        _target = target_description[basic_index+28:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' can be understood as ' in target_description:
        basic_index = target_description.index(' can be understood as ')
        _target = target_description[basic_index+22:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' might be ' in target_description:
        basic_index = target_description.index(' might be ')
        _target = target_description[basic_index+10:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    elif ' may be ' in target_description:
        basic_index = target_description.index(' may be ')
        _target = target_description[basic_index+8:]
        _period_index = _target.index('.')
        final_target = _target[:_period_index]
    else:
        print('check target meaning', target_description)
        final_target = target_description
    if final_target[0] == "'" and final_target[-1] == "'":
        final_target = final_target.strip("'")
    elif final_target[0] == '"' and final_target[-1] == '"':
        final_target = final_target.strip('"')
    if final_target[:4] == 'the ':
        final_target = final_target[4:]
    if len(final_target) >= 17:
        if final_target[:17] == 'target domain of ':
            final_target = final_target[17:]
    if len(final_target) >= 10:
        if final_target[:10] == 'domain of ':
            final_target = final_target[10:]
    if len(final_target) >= 20:
        if final_target[:20] == 'in this sentence is ':
            final_target = final_target[20:]
    return final_target

def get_source(source_description):
    # extract basic meaning

    if source_description[-1] != '.':
        source_description = source_description + '.'
    if ' is typically the domain of ' in source_description:
        basic_index = source_description.index(' is typically the domain of ')
        _source = source_description[basic_index+28:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' is the domain of ' in source_description:
        basic_index = source_description.index(' is the domain of ')
        _source = source_description[basic_index+18:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' implies the source domain of ' in source_description:
        basic_index = source_description.index(' implies the source domain of ')
        _source = source_description[basic_index+30:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' typically implies the source domain of ' in source_description:
        basic_index = source_description.index(' typically implies the source domain of ')
        _source = source_description[basic_index+40:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' typically refers to ' in source_description:
        basic_index = source_description.index(' typically refers to ')
        _source = source_description[basic_index+21:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]  
    elif ' typically relates to ' in source_description:
        basic_index = source_description.index(' typically relates to ')
        _source = source_description[basic_index+22:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' is typically related to ' in source_description:
        basic_index = source_description.index(' is typically related to ')
        _source = source_description[basic_index+25:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]   
    elif ' typically associated with the domain of ' in source_description:
        basic_index = source_description.index(' typically associated with the domain of ')
        _source = source_description[basic_index+41:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' associated with ' in source_description:
        basic_index = source_description.index(' associated with ')
        _source = source_description[basic_index+17:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' is related to ' in source_description:
        basic_index = source_description.index(' is related to ')
        _source = source_description[basic_index+15:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' suggests ' in source_description:
        basic_index = source_description.index(' suggests ')
        _source = source_description[basic_index+10:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]  
    elif ' is the ' in source_description:
        basic_index = source_description.index(' is the')
        _source = source_description[basic_index+8:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]    
    elif ' is likely related to ' in source_description:
        basic_index = source_description.index(' is likely related to')
        _source = source_description[basic_index+22:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index] 
    elif ' is likely the domain of ' in source_description:
        basic_index = source_description.index(' is likely the domain of ')
        _source = source_description[basic_index+25:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' is likely the ' in source_description:
        basic_index = source_description.index(' is likely the ')
        _source = source_description[basic_index+15:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif '" is ' in source_description:
        basic_index = source_description.index('" is ')
        _source = source_description[basic_index+5:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' is that of ' in source_description:
        basic_index = source_description.index(' is that of ')
        _source = source_description[basic_index+12:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' tries to describe ' in source_description:
        basic_index = source_description.index(' tries to describe ')
        _source = source_description[basic_index+19:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' describe the ' in source_description:
        basic_index = source_description.index(' describe the ')
        _source = source_description[basic_index+14:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' it describes ' in source_description:
        basic_index = source_description.index(' it describes ')
        _source = source_description[basic_index+14:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' describes the ' in source_description:
        basic_index = source_description.index(' describes the ')
        _source = source_description[basic_index+15:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' be related to ' in source_description:
        basic_index = source_description.index(' be related to')
        _source = source_description[basic_index+15:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' related to ' in source_description:
        basic_index = source_description.index(' related to ')
        _source = source_description[basic_index+12:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' refers to ' in source_description:
        basic_index = source_description.index(' refers to ')
        _source = source_description[basic_index+11:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' implies' in source_description:
        basic_index = source_description.index(' implies ')
        _source = source_description[basic_index+9:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' involves ' in source_description:
        basic_index = source_description.index(' involves ')
        _source = source_description[basic_index+10:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' pertains to ' in source_description:
        basic_index = source_description.index(' pertains to ')
        _source = source_description[basic_index+13:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' would likely be ' in source_description:
        basic_index = source_description.index(' would likely be ')
        _source = source_description[basic_index+17:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' would related to ' in source_description:
        basic_index = source_description.index(' would related to ')
        _source = source_description[basic_index+18:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' would be ' in source_description:
        basic_index = source_description.index(' would be ')
        _source = source_description[basic_index+10:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' could be ' in source_description:
        basic_index = source_description.index(' could be ')
        _source = source_description[basic_index+10:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' can be ' in source_description:
        basic_index = source_description.index(' can be ')
        _source = source_description[basic_index+8:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]        
    elif ' would generally be ' in source_description:
        basic_index = source_description.index(' would generally be ')
        _source = source_description[basic_index+20:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' would typically be ' in source_description:
        basic_index = source_description.index(' would typically be ')
        _source = source_description[basic_index+20:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' can be understood as ' in source_description:
        basic_index = source_description.index(' can be understood as ')
        _source = source_description[basic_index+22:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' is ' in source_description:
        basic_index = source_description.index(' is ')
        _source = source_description[basic_index+4:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' relates to ' in source_description:
        basic_index = source_description.index(' relates to ')
        _source = source_description[basic_index+12:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' generally belongs to ' in source_description:
        basic_index = source_description.index(' generally belongs to ')
        _source = source_description[basic_index+22:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]        
    elif ' could be that of ' in source_description:
        basic_index = source_description.index(' could be that of ')
        _source = source_description[basic_index+18:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' generally corresponds to ' in source_description:
        basic_index = source_description.index(' generally corresponds to ')
        _source = source_description[basic_index+26:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' might be ' in source_description:
        basic_index = source_description.index(' might be ')
        _source = source_description[basic_index+10:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    elif ' may be ' in source_description:
        basic_index = source_description.index(' may be ')
        _source = source_description[basic_index+8:]
        _period_index = _source.index('.')
        final_source = _source[:_period_index]
    else:
        print('check source meaning', source_description)
        final_source = source_description
    if final_source == '':
        return ''
    if final_source[0] == "'" and final_source[-1] == "'":
        final_source = final_source.strip("'")
    elif final_source[0] == '"' and final_source[-1] == '"':
        final_source = final_source.strip('"')
    if final_source[:4] == 'the ':
        final_source = final_source[4:]
    if len(final_source) >= 17:
        if final_source[:17] == 'source domain of ':
            final_source = final_source[17:]
    if len(final_source) >= 10:
        if final_source[:10] == 'domain of ':
            final_source = final_source[10:]
    if len(final_source) >= 20:
        if final_source[:20] == 'in this sentence is ':
            final_source = final_source[20:]
    return final_source.strip('"').strip("'")

def main(args):
    args.output_method = args.method + '-extract'
    dataloader = Data_Processor(args)
    new_samples = []
    print(args.data_dir, args.method, args.model_dir, args.dataset)
    for file in dataset_to_fileList[args.dataset]:
        samples = dataloader.get_dataset(file)
        for i, sample in enumerate(samples):
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text = sample
            answers = eval(answer_text)
            source_extract_list = []
            target_extract_list = []
            for source_ori in answers[0]:
                if source_ori == '':
                    source_extract_list.append(source_ori)
                else:
                    source_extract = get_source(source_ori)
                    source_extract_list.append(source_extract)
            for target_ori in answers[1]:
                if target_ori == '':
                    target_extract_list.append(target_ori)
                else:
                    target_extract = get_target(target_ori)
                    target_extract_list.append(target_extract)

            new_samples.append(sample + [str(source_extract_list)] + [str(target_extract_list)])
        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer2_extract']
        dataloader.write_dataset(file, head, new_samples)    
            

if __name__ == '__main__':
    args = parse_option()
    
    main(args)
