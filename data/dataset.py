#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Dataset for TypoTransformer
#

import os
import re
import csv
import torch
import multiprocessing
import random
import json
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader, IterableDataset

from utils.mimic_tools import MIMICPseudonymizer
from scripts.utils import sanitize_text, clean_text

from fastDamerauLevenshtein import damerauLevenshtein

DEFAULT_MAX_CHARACTER_POSITIONS = 64

# char_tokens = list("0123456789abcdefghijklmnopqrstuvwxyz+-*/^.,;:=!?'()[]{}")
char_tokens = list("0123456789abcdefghijklmnopqrstuvwxyz+-*/^.,;:=!?'()[]{}&")


class TypoDataset(Dataset):
    """
    This dataset gives examples of (typo, context, correction)
    """
    def __init__(self, tsv_file, bert_tokenizer, typo_tokenizer, num_process=None):
        assert os.path.exists(tsv_file), f'{tsv_file} does not exists'

        self.tsv_file = tsv_file
        self.bert_tokenizer = bert_tokenizer
        self.typo_tokenizer = typo_tokenizer

        print(f'Read file {tsv_file}... ', end='', flush=True)
        self.csv_rows = []
        with open(self.tsv_file, 'r') as fd:
            reader = csv.reader(fd, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0: continue
                self.csv_rows.append(row)
        print(f'{len(self.csv_rows)} rows')

        num_process = num_process if num_process is not None else multiprocessing.cpu_count()
        num_process = min(num_process, len(self.csv_rows))
        print(f'Parsing rows ({num_process} processes)')
        pool = multiprocessing.Pool(num_process)
        self.examples = []
        for example in tqdm(pool.imap(self._parse_row, self.csv_rows, chunksize=len(self.csv_rows)//num_process),
                            total=len(self.csv_rows)):
            self.examples.append(example)

    def _make_sentence(self, tokens_left, tokens_right, seq_length=128):
        len_left = len(tokens_left)
        len_right = len(tokens_right)

        cut_len = len_left + len_right - (seq_length - 1)
        if cut_len > 0:
            cut_left = len_left - seq_length // 2
            cut_right = len_right - (seq_length-1) // 2
            if cut_left < 0:
                cut_left, cut_right = 0, cut_left + cut_right
            elif cut_right < 0:
                cut_left, cut_right = cut_left + cut_right, 0
        else:
            cut_left, cut_right = 0, 0

        tokens_left = tokens_left[cut_left:]
        # tokens_right = tokens_right[:-cut_right]
        tokens_right = tokens_right[:len(tokens_right)-cut_right]

        tokens = tokens_left + [self.bert_tokenizer.mask_token] + tokens_right
        attention_mask = [1] * len(tokens_left) + [1] + [1] * len(tokens_right)

        if len(tokens) < seq_length:
            num_padding = seq_length - len(tokens)
            tokens += [self.bert_tokenizer.pad_token] * num_padding
            attention_mask += [0] * num_padding
        return tokens, attention_mask

    def _parse_row(self, row):
        """
        Convert a csv row to examples
        """
        ex_id, note_id, typo, left, right, correct = row

        tokens_left = self.bert_tokenizer.tokenize(left)
        tokens_right = self.bert_tokenizer.tokenize(right)
        context_tokens, context_attention_mask = self._make_sentence(tokens_left, tokens_right)
        context_token_ids = self.bert_tokenizer.convert_tokens_to_ids(context_tokens)

        typo_token_ids, typo_attention_mask = self.typo_tokenizer.tokenize(
            typo, eos_bos=True, padding_end=False, output_token_ids=True)
        correct_token_ids, correct_attention_mask = self.typo_tokenizer.tokenize(
            correct, eos_bos=True, padding_end=False, output_token_ids=True)

        example = {'example_id': int(ex_id),
                   'note_id': int(note_id),
                   'context_tokens': context_token_ids,
                   'context_attention_mask': context_attention_mask,
                   'typo': typo,
                   'typo_token_ids': typo_token_ids,
                   'typo_attention_mask': typo_attention_mask,
                   'correct': correct,
                   'correct_token_ids': correct_token_ids,
                   'correct_attention_mask': correct_attention_mask}

        return example

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def get_collate_fn(self, stack=True):
        def _collate_fn(examples):
            max_typo_len = max([len(e['typo_token_ids']) for e in examples])
            max_correct_len = max([len(e['correct_token_ids']) for e in examples])
            max_typo_len = max_correct_len = max(max_typo_len, max_correct_len) + 1
            for e in examples:
                num_typo_padding = max_typo_len - len(e['typo_token_ids'])
                e['typo_token_ids'] = e['typo_token_ids'] + [self.typo_tokenizer.pad_index] * num_typo_padding
                e['typo_attention_mask'] = e['typo_attention_mask'] + [0] * num_typo_padding
                num_correct_padding = max_correct_len - len(e['correct_token_ids'])
                e['correct_token_ids'] = e['correct_token_ids'] + [self.typo_tokenizer.pad_index] * num_correct_padding
                e['correct_attention_mask'] = e['correct_attention_mask'] + [0] * num_correct_padding
            if stack:
                batch = {}
                for k in examples[0].keys():
                    batch[k] = [e[k] for e in examples]
                    if k not in ['typo', 'correct']:
                        batch[k] = torch.tensor(batch[k])
                return batch
            else:
                return examples
        return _collate_fn


class TypoOnlineDataset(TypoDataset, IterableDataset):
    def __init__(self, csv_dir, dict_file, bert_tokenizer, typo_tokenizer,
                 max_word_corruptions=2, do_substitution=True, do_transposition=True,
                 no_corruption_prob=0.0, min_word_len=3):
        # Not call TypoDataset init()
        IterableDataset.__init__(self)

        self.csv_dir = csv_dir
        self.csv_fnames = [f for f in os.listdir(self.csv_dir) \
                           if f.startswith('NOTEEVENTS') and f.endswith('.csv')]
        self.dict_file = dict_file
        with open(self.dict_file, 'r') as fd:
            self.dictionary = set(json.load(fd))
        self.bert_tokenizer = bert_tokenizer
        self.typo_tokenizer = typo_tokenizer

        self.max_word_corruptions = max_word_corruptions
        self.do_substitution = do_substitution
        self.do_transposition = do_transposition
        self.word_corrupter = WordCorrupter(self.max_word_corruptions, self.do_substitution,
                                            self.do_transposition)
        self.no_corruption_prob = no_corruption_prob
        self.min_word_len = min_word_len

        mimic_tools_dir = 'scripts/mimic-tools/lists'
        self.mimic_pseudo = MIMICPseudonymizer(mimic_tools_dir)


    def _load_random_csv(self):
        self.csv_fname = random.choice(self.csv_fnames)
        self.csv_fpath = os.path.join(self.csv_dir, self.csv_fname)

        self.df_note = pd.read_csv(self.csv_fpath, low_memory=False)
        self.note_iterrows = self.df_note.iterrows()

    def _random_word_context(self, text, max_trial=10):
        puncs = list("[]!\"#$%&'()*+,./:;<=>?@\^_`{|}~-")
        words = text.split()

        trial = 0
        done = False
        while trial < max_trial and not done:
            trial += 1
            w_idx = random.randint(0, len(words)-1)
            word, left_res, right_res = words[w_idx], [], []

            # If the word is already in vocab, it's good to go.
            if len(word) >= self.min_word_len and \
                    (word.lower() in self.dictionary) and \
                    len(word) < DEFAULT_MAX_CHARACTER_POSITIONS - 4:
                done = True
            else:
                # Otherwise, detach puncs at the first and the last char, and check again
                if word[0] in puncs:
                    word, left_res = word[1:], [word[0]]
                else:
                    word, left_res = word, []
                if not word: continue  # The word was just a punc

                if word[-1] in puncs:
                    word, right_res = word[:-1], [word[-1]]
                else:
                    word, right_res = word, []

                if len(word) < self.min_word_len or \
                        (not word.lower() in self.dictionary) or \
                        len(word) >= DEFAULT_MAX_CHARACTER_POSITIONS - 4:
                    continue

                # Check whether it's anonymized field
                right_snip = ' '.join(words[w_idx+1:w_idx+5])
                if '**]' in right_snip and '[**' not in right_snip:
                    continue
                left_snip = ' '.join(words[w_idx-4:w_idx])
                if '[**' in left_snip and '**]' not in left_snip:
                    continue

                # Pass!
                done = True

        if done:
            return word, ' '.join(words[:w_idx] + left_res), ' '.join(right_res + words[w_idx+1:])
        else:
            raise ValueError('failed to choose word')

    def _process_note(self, note):
        note = re.sub('\n', ' ', note)
        note = re.sub('\t', ' ', note)
        note = sanitize_text(clean_text(note))
        return note

    def __iter__(self):
        self._load_random_csv()
        return self

    def __next__(self):
        # Select next note (length >= 2000)
        while True:
            try:
                _, row = next(self.note_iterrows)
            except StopIteration:
                self._load_random_csv()
                _, row = next(self.note_iterrows)
            note_id = int(row.ROW_ID)
            note = row.TEXT.strip()
            # if len(note) >= 2000:
                # break
            if len(note) < 2000:
                continue

            try:
                correct, left, right = self._random_word_context(note)
            except:
                # import traceback; traceback.print_exc();
                continue
            break

        # Corrupt and pseudonymize
        correct = correct.lower()
        if random.uniform(0, 1) >= self.no_corruption_prob:
            typo = self.word_corrupter.corrupt_word(correct)
        else:
            typo = correct
        left = self.mimic_pseudo.pseudonymize(left)
        left = self._process_note(left)
        left = ' '.join(left.split(' ')[-128:])
        right = self.mimic_pseudo.pseudonymize(right)
        right = self._process_note(right)
        right = ' '.join(right.split(' ')[:128])

        # Parse
        temp_csv_row = [-1, note_id, typo, left, right, correct]
        # print(f'{self.csv_fname}({note_id}, {_}/{len(self.df_note)}): {correct} -> {typo}')
        example = self._parse_row(temp_csv_row)

        return example

class WordCorrupter(object):
    def __init__(self, max_corruptions=2, do_substitution=True, do_transposition=True):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.max_corruptions = max_corruptions
        self.do_substitution = do_substitution
        self.do_transposition = do_transposition
        self.operation_list = ['ins', 'del']
        if self.do_substitution:
            self.operation_list.append('sub')
        if do_transposition:
            self.operation_list.append('tra')

    def random_alphabet(self):
        return random.choice(self.alphabet)

    def single_corruption(self, word):
        while True:
            oper = random.choice(self.operation_list)

            if oper == "del":  # deletion
                if len(word) == 1: continue
                cidx = random.randint(0, len(word)-1)
                ret = word[:cidx] + word[cidx+1:]
                break
            elif oper == "ins":  # insertion
                cidx = random.randint(0, len(word))
                ret = word[:cidx] + self.random_alphabet() + word[cidx:]
                break
            elif oper == "sub":  # substitution
                cidx = random.randint(0, len(word)-1)
                while True:
                    c = self.random_alphabet()
                    if c != word[cidx]:
                        ret = word[:cidx] + c + word[cidx+1:]
                        break
            elif oper == "tra":  # transposition
                if len(word) == 1 : continue
                cidx = random.randint(0, len(word)-2) # swap cidx-th and (cidx+1)-th char
                if word[cidx+1] == word[cidx]: continue
                ret = word[:cidx] + word[cidx+1] + word[cidx] + word[cidx+2:]
                break
            else:
                raise ValueError(f'Wrong operation {oper}')
        return ret

    def corrupt_word(self, word_original):
        num_corruption = random.randint(1, self.max_corruptions)
        while True:
            word = word_original
            for i in range(num_corruption):
                word = self.single_corruption(word)
            if word_original != word:
                break
        return word
