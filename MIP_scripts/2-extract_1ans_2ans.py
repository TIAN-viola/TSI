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
    parser.add_argument('--ckpt_dir', default='gpt-3.5-turbo-0613', type=str, help='gpt-3.5-turbo-0613')
    parser.add_argument('--baseline', default='MIP-1que-2que-1.0', type=str, help='')
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

def get_basic_meaning(basic_meaning_ori, target_word):
    # extract basic meaning
    if 'inct, or difficult to see or understand clearly' in basic_meaning_ori:
        print(basic_meaning_ori)
    basic_meaning_ori = basic_meaning_ori.lower()
    if 'a common meaning is' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('a common meaning is')
        _basic_meaning = basic_meaning_ori[basic_index+20:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'the basic meaning of' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('the basic meaning of')
        _basic_meaning = basic_meaning_ori[basic_index+21:]
        
        if '1.' in _basic_meaning:
            _basic_index = _basic_meaning.index('1.')
            _basic_meaning = _basic_meaning[_basic_index+3:]

        if '."' in _basic_meaning:
            _quoate_index = _basic_meaning.index('."')
            basic_meaning_quoat = _basic_meaning[_quoate_index+2:]
            if '.' in basic_meaning_quoat:
                _period_index = basic_meaning_quoat.index('.')
                __basic_meaning = _basic_meaning[:_quoate_index+_period_index+2]
            else:
                __basic_meaning = _basic_meaning
        else:
            if '.' in _basic_meaning:
                _period_index = _basic_meaning.index('.')
                __basic_meaning = _basic_meaning[:_period_index]
            else:
                __basic_meaning = _basic_meaning
        if ' is that ' in __basic_meaning:
            _basic_index = __basic_meaning.index(' is that ')
            final_basic_meaning = __basic_meaning[_basic_index+9:]
        elif ' is to ' in __basic_meaning:
            _basic_index = __basic_meaning.index(' is to ')
            final_basic_meaning = __basic_meaning[_basic_index+7:]
        elif ' is ' in __basic_meaning:
            _basic_index = __basic_meaning.index(' is ')
            final_basic_meaning = __basic_meaning[_basic_index+4:]
        elif ' refers to ' in __basic_meaning:
            _basic_index = __basic_meaning.index(' refers to ')
            final_basic_meaning = __basic_meaning[_basic_index+11:]
        elif ' refer to ' in __basic_meaning:
            _basic_index = __basic_meaning.index(' refer to ')
            final_basic_meaning = __basic_meaning[_basic_index+10:]
        elif ' relates to ' in __basic_meaning:
            _basic_index = __basic_meaning.index(' relates to ')
            final_basic_meaning = __basic_meaning[_basic_index+12:]   
        elif ' can be defined as ' in __basic_meaning:
            _basic_index = __basic_meaning.index(' can be defined as ')
            final_basic_meaning = __basic_meaning[_basic_index+18:] 
        elif 'verb form: ' in __basic_meaning:
            _basic_index = __basic_meaning.index('verb form: ')
            _final_basic_meaning = __basic_meaning[_basic_index+11:]
            if ' refer to ' in _final_basic_meaning:
                __basic_index = _final_basic_meaning.index(' refer to ')
                _final_basic_meaning = _final_basic_meaning[__basic_index+10:]
            else:
                _final_basic_meaning = _final_basic_meaning
        elif '1. ' in __basic_meaning:
            _basic_index = __basic_meaning.index('1.')
            final_basic_meaning = __basic_meaning[_basic_index+3:]
        elif 'is: ' in __basic_meaning:
            _basic_index = __basic_meaning.index('is: ')
            final_basic_meaning = __basic_meaning[_basic_index+4:]
        elif 'suggests ' in __basic_meaning:
            _basic_index = __basic_meaning.index('suggests ')
            final_basic_meaning = __basic_meaning[_basic_index+9:]
        elif 'refers to ' in __basic_meaning:
            _basic_index = __basic_meaning.index('refers to ')
            final_basic_meaning = __basic_meaning[_basic_index+10:]
        elif 'describes ' in __basic_meaning:
            _basic_index = __basic_meaning.index('describes ')
            final_basic_meaning = __basic_meaning[_basic_index+10:]
        else:
            print('check basic meaning', basic_meaning_ori)
            final_basic_meaning = __basic_meaning

    elif 'the word "%s" means that' %target_word in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('the word "%s" means that' %target_word)
        _basic_meaning = basic_meaning_ori[basic_index+len('the word "%s" means that' %target_word)+1:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif ' means that' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' means that')
        _basic_meaning = basic_meaning_ori[basic_index+12:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif ' means ' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' means ')
        _basic_meaning = basic_meaning_ori[basic_index+7:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]  
    elif ' refers to ' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' refers to ')
        _basic_meaning = basic_meaning_ori[basic_index+11:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' refer to ' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' refer to ')
        _basic_meaning = basic_meaning_ori[basic_index+11:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' to mean ' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' to mean ')
        _basic_meaning = basic_meaning_ori[basic_index+9:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif ' to imply ' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' to imply ')
        _basic_meaning = basic_meaning_ori[basic_index+10:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'its basic meaning here is' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('its basic meaning here is')
        _basic_meaning = basic_meaning_ori[basic_index+26:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'it suggests that' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('it suggests that')
        _basic_meaning = basic_meaning_ori[basic_index+17:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'it suggests' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('it suggests')
        _basic_meaning = basic_meaning_ori[basic_index+12:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'it implies that' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('it implies that')
        _basic_meaning = basic_meaning_ori[basic_index+16:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'it implies' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('it implies')
        _basic_meaning = basic_meaning_ori[basic_index+11:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'implies that' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('implies that')
        _basic_meaning = basic_meaning_ori[basic_index+13:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif 'implies' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index('implies')
        _basic_meaning = basic_meaning_ori[basic_index+8:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]
    elif ' is to' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' is to')
        _basic_meaning = basic_meaning_ori[basic_index+7:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index]    
    elif ' describes' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' describes')
        _basic_meaning = basic_meaning_ori[basic_index+11:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' is used to describe' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' is used to describe')
        _basic_meaning = basic_meaning_ori[basic_index+22:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' is that' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' is that')
        _basic_meaning = basic_meaning_ori[basic_index+9:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' indicates that' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' indicates that')
        _basic_meaning = basic_meaning_ori[basic_index+16:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' refers to' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' refers to')
        _basic_meaning = basic_meaning_ori[basic_index+11:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' is typically used as' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' is typically used as')
        _basic_meaning = basic_meaning_ori[basic_index+22:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    elif ' describes ' in basic_meaning_ori:
        basic_index = basic_meaning_ori.index(' describes ')
        _basic_meaning = basic_meaning_ori[basic_index+11:]
        _period_index = _basic_meaning.index('.')
        final_basic_meaning = _basic_meaning[:_period_index] 
    
    
    else:
        print('check basic meaning', basic_meaning_ori)
        final_basic_meaning = basic_meaning_ori
    return final_basic_meaning

def get_contextual_meaning(contextual_meaning_ori):
    contextual_meaning_ori = contextual_meaning_ori.lower()
    if 'implies that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('implies that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+13:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]
    elif 'implies' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('implies')
        _contextual_meaning = contextual_meaning_ori[contextual_index+8:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]
    elif ' means ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' means ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+7:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]    
    elif 'the contextual meaning of' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('the contextual meaning of')
        _contextual_meaning = contextual_meaning_ori[contextual_index+26:]
        if '1.' in _contextual_meaning:
            _contextual_index = _contextual_meaning.index('1.')
            _contextual_meaning = _contextual_meaning[_contextual_index+3:]

        if '."' in _contextual_meaning:
            _quoate_index = _contextual_meaning.index('."')
            contextual_meaning_quoat = _contextual_meaning[_quoate_index+2:]
            if '.' in contextual_meaning_quoat:
                _period_index = contextual_meaning_quoat.index('.')
                __contextual_meaning = _contextual_meaning[:_quoate_index+_period_index+2]
            else:
                __contextual_meaning = _contextual_meaning
        else:
            if '.' in _contextual_meaning:
                _period_index = _contextual_meaning.index('.')
                __contextual_meaning = _contextual_meaning[:_period_index]
            else:
                __contextual_meaning = _contextual_meaning
        if 'is that' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('is that')
            final_contextual_meaning = __contextual_meaning[_contextual_index+8:]
        elif ' is to ' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index(' is to ')
            final_contextual_meaning = __contextual_meaning[_contextual_index+7:]
        elif ' is ' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index(' is ')
            final_contextual_meaning = __contextual_meaning[_contextual_index+4:]
        elif ' refers to ' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index(' refers to ')
            final_contextual_meaning = __contextual_meaning[_contextual_index+11:]
        elif 'indicates that' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('indicates that')
            final_contextual_meaning = __contextual_meaning[_contextual_index+15:]
        elif 'indicates' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('indicates')
            final_contextual_meaning = __contextual_meaning[_contextual_index+10:]
        elif 'suggests that' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('suggests that')
            final_contextual_meaning = __contextual_meaning[_contextual_index+14:]    
        elif 'implies that' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('implies that')
            final_contextual_meaning = __contextual_meaning[_contextual_index+13:]  
        elif 'means to' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('means to')
            final_contextual_meaning = __contextual_meaning[_contextual_index+9:]
        elif 'would be to' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('would be to')
            final_contextual_meaning = __contextual_meaning[_contextual_index+12:]    
        elif 'highlights' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('highlights')
            final_contextual_meaning = __contextual_meaning[_contextual_index+12:]  
        elif 'describes' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('describes')
            final_contextual_meaning = __contextual_meaning[_contextual_index+11:] 
        elif 'indicates that' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('indicates that')
            final_contextual_meaning = __contextual_meaning[_contextual_index+15:] 
        elif 'suggest' in __contextual_meaning:
            _contextual_index = __contextual_meaning.index('suggest')
            final_contextual_meaning = __contextual_meaning[_contextual_index+8:] 
        elif 'carries the contextual meaning of' in contextual_meaning_ori:
            final_contextual_meaning = __contextual_meaning
        elif 'has the contextual meaning of' in contextual_meaning_ori:
            final_contextual_meaning = __contextual_meaning
        else:
            print('check contextual meaning', contextual_meaning_ori)
            final_contextual_meaning = contextual_meaning_ori

    elif ' refers to ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' refers to ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+11:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]
    elif 'imply that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('imply that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+11:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]  
    elif ' to imply' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' to imply')
        _contextual_meaning = contextual_meaning_ori[contextual_index+10:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' imply' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' imply')
        _contextual_meaning = contextual_meaning_ori[contextual_index+7:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' refers to ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' refers to ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+11:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]  
    elif 'means that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('means that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+11:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]   
    elif ' to mean ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' to mean ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+9:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' means to ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' means to ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+10:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' means ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' means ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+7:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' meaning ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' meaning ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+9:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'it suggests that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('it suggests that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+17:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'it suggests' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('it suggests')
        _contextual_meaning = contextual_meaning_ori[contextual_index+12:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'suggests that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('suggests that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+14:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]     
    elif 'convey that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('convey that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+12:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' refer to ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' refer to ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+10:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'indicates that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('indicates that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+15:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'indicates' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('indicates')
        _contextual_meaning = contextual_meaning_ori[contextual_index+10:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'could mean that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('could mean that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+16:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'conveys the idea that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('conveys the idea that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+22:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' to describe' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' to describe')
        _contextual_meaning = contextual_meaning_ori[contextual_index+13:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'to convey that' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('to convey that')
        _contextual_meaning = contextual_meaning_ori[contextual_index+15:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]     
    elif 'to convey' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('to convey')
        _contextual_meaning = contextual_meaning_ori[contextual_index+10:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]   
    elif 'to express' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('to express')
        _contextual_meaning = contextual_meaning_ori[contextual_index+11:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index]    
    elif ' is used to' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' is used to')
        _contextual_meaning = contextual_meaning_ori[contextual_index+12:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif 'it denotes' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index('it denotes')
        _contextual_meaning = contextual_meaning_ori[contextual_index+10:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' is describing ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' is describing ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+15:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' describes ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' describes ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+11:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' means ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' means ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+7:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' describe that ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' describe that ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+15:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' indicate that ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' indicate that ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+15:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' indicating that ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' indicating that ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+17:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' reference to represent ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' reference to represent ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+24:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' is used as ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' is used as ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+12:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' is likely referring to ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' is likely referring to ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+24:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' suggest ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' suggest ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+9:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' suggests ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' suggests ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+10:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' signifies that ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' signifies that ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+16:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' conveys ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' conveys ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+9:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' is a verb that ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' is a verb that ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+16:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    elif ' referring to ' in contextual_meaning_ori:
        contextual_index = contextual_meaning_ori.index(' referring to ')
        _contextual_meaning = contextual_meaning_ori[contextual_index+14:]
        _period_index = _contextual_meaning.index('.')
        final_contextual_meaning = _contextual_meaning[:_period_index] 
    else:
        print("contextual_meaning_ori", contextual_meaning_ori)
        final_contextual_meaning = contextual_meaning_ori
    return final_contextual_meaning

def main(args):
    args.output_baseline = args.baseline + '-extract'
    dataloader = Data_Processor(args)
    
    new_samples = []
    print(args.data_dir, args.baseline, args.ckpt_dir, args.dataset)
    for file in dataset_to_fileList[args.dataset]:
        samples = dataloader.get_dataset(file)
        for i, sample in enumerate(samples):
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,prompt,answer_text,answer_add_text = sample
            answers = eval(answer_text)
            basic_meaning_extract_list = []
            contextual_meaning_extract_list = []
            for basic_meaning_ori in answers[0]:
                basic_meaning_extract = get_basic_meaning(basic_meaning_ori, target_word)
                basic_meaning_extract_list.append(basic_meaning_extract)
            for contextual_meaning_ori in answers[1]:
                contextual_meaning_extract = get_contextual_meaning(contextual_meaning_ori)
                contextual_meaning_extract_list.append(contextual_meaning_extract)

            new_samples.append(sample + [str(basic_meaning_extract_list)] + [str(contextual_meaning_extract_list)])
        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg', 'answer1_extract', 'answer2_extract']
        dataloader.write_dataset(file, head, new_samples)    
            
if __name__ == '__main__':
    args = parse_option()

    main(args)
