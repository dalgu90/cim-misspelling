#!/usr/bin/env python
# -*- coding=utf-8 -*-

#
# Noisy Channel Model Typo Correction
# Author: Juyong Kim
#

import os
import sys
import json
import argparse
import time
import pickle
import random
from datetime import datetime
import numpy as np
import torch
import fairseq
import transformers
import types
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from fastDamerauLevenshtein import damerauLevenshtein

from data.dataset import TypoDataset, TypoOnlineDataset
from model.char_lm import CharacterLanguageModel, CharTokenizer, CrossEntropyLoss, Trie
from model.parallel import DataParallelModel, DataParallelCriterion
from utils.checkpoint_manager import CheckPointManager

parser = argparse.ArgumentParser(description='Typo correction training script')
parser.add_argument("--mimic_csv_dir", type=str, default="data/mimic3/split")
parser.add_argument("--data_dir", type=str, default="data/mimic_synthetic")
parser.add_argument("--dict_file", type=str, default="data/lexicon/lexicon_en.json")
parser.add_argument("--bert_dir", type=str, default="bert/ncbi_bert_base")
parser.add_argument("--output_dir", type=str, default="results/cim_base")
parser.add_argument('--is_train', dest='is_train', action='store_true')
parser.add_argument('--is_eval', dest='is_train', action='store_false')
parser.set_defaults(is_train=False)
parser.add_argument("--test_file", type=str, default="test.tsv")
parser.add_argument("--init_ckpt", type=str, default=None)
parser.add_argument("--init_step", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)
# Model parameter
parser.add_argument("--decoder_layers", type=int, default=12)
parser.add_argument('--train_bert', dest='train_bert', action='store_true')
parser.add_argument('--no_train_bert', dest='train_bert', action='store_false')
parser.set_defaults(train_bert=False)
parser.add_argument("--bert_finetune_factor", type=float, default=1.0)
parser.add_argument("--dropout", type=float, default=0.1)
# Synthetic dataset parameter
parser.add_argument("--synthetic_min_word_len", type=int, default=3)
parser.add_argument('--do_substitution', dest='do_substitution', action='store_true')
parser.add_argument('--do_not_substitution', dest='do_substitution', action='store_false')
parser.set_defaults(do_substitution=True)
parser.add_argument('--do_transposition', dest='do_transposition', action='store_true')
parser.add_argument('--do_not_transposition', dest='do_transposition', action='store_false')
parser.set_defaults(do_transposition=True)
parser.add_argument("--max_word_corruptions", type=int, default=2)
parser.add_argument("--no_corruption_prob", type=float, default=0.0)
# Optimization parameter
parser.add_argument('--train_with_ed', dest='train_with_ed', action='store_true')
parser.add_argument('--no_train_with_ed', dest='train_with_ed', action='store_false')
parser.set_defaults(train_with_ed=False)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--adam_betas", type=str, default='(0.9, 0.999)')
parser.add_argument("--adam_eps", type=float, default=1e-08)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--training_step", type=int, default=200000)
parser.add_argument("--display_iter", type=int, default=25)
parser.add_argument("--eval_iter", type=int, default=2000)
parser.add_argument("--lr_scheduler", type=str, default='inverse_sqrt')
parser.add_argument("--lr", type=float, nargs='+', default=[0.0001])
parser.add_argument("--warmup_updates", type=int, default=10000)
parser.add_argument("--warmup_init_lr", type=float, default=1e-6)
# Evaluation parameter
parser.add_argument("--num_beams", type=int, default=5)
parser.add_argument("--edit_distance_weight", type=float, default=1.0)
parser.add_argument("--edit_distance_extra_len", type=int, default=100)  # no effect if it's too large
parser.add_argument("--length_penalty", type=float, default=1.0)
parser.add_argument('--beam_sort_linear_ed', dest='beam_sort_linear_ed', action='store_true')
parser.add_argument('--beam_sort_ed', dest='beam_sort_linear_ed', action='store_false')
parser.set_defaults(beam_sort_linear_ed=False)
parser.add_argument('--beam_final_score_normalize_ed', dest='beam_final_score_normalize_ed', action='store_true')
parser.add_argument('--beam_final_score_unnormalize_ed', dest='beam_final_score_normalize_ed', action='store_false')
parser.set_defaults(beam_final_score_normalize_ed=False)
parser.add_argument('--dict_matching', dest='dict_matching', action='store_true')
parser.add_argument('--no_dict_matching', dest='dict_matching', action='store_false')
parser.set_defaults(dict_matching=False)
args = parser.parse_args()

# Print parameters
print('\n'.join([f'\t{k}: {v}' for k, v in vars(args).items()]))


def main():
    # set flags / seeds
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # BERT and LevT configs
    bert_config_file = os.path.join(args.bert_dir, 'bert_config.json')
    bert_config = CharacterLanguageModel.get_bert_config(bert_config_file)
    char_decoder_config = CharacterLanguageModel.get_char_decoder_config(
            bert_config_file, args)

    # BERT tokenizer and LevT tokenizer
    bert_vocab_file = os.path.join(args.bert_dir, 'vocab.txt')
    bert_tokenizer = CharacterLanguageModel.get_bert_tokenizer(bert_vocab_file)
    typo_tokenizer = CharTokenizer()

    # Trie for dictionary matching
    if args.dict_matching:
        trie = Trie(typo_tokenizer, eos_token_id=typo_tokenizer.eos_index)
        with open(args.dict_file, 'r') as fd:
            dict_words = json.load(fd)
        trie.add_words(dict_words)
    else:
        trie = None

    # Dataset
    if args.is_train:
        dataset_train = TypoOnlineDataset(args.mimic_csv_dir, args.dict_file, bert_tokenizer, typo_tokenizer,
                                          args.max_word_corruptions, args.do_substitution, args.do_transposition,
                                          args.no_corruption_prob, args.synthetic_min_word_len,
                                          args.train_with_ed, args.edit_distance_extra_len)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=args.batch_size,
                                                       num_workers=4*args.num_gpus,
                                                       collate_fn=dataset_train.get_collate_fn())
        dataset_val = TypoDataset(os.path.join(args.data_dir, 'val.tsv'), bert_tokenizer, typo_tokenizer)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     drop_last=True,
                                                     num_workers=1,
                                                     collate_fn=dataset_val.get_collate_fn())
    else:  # args.is_train == False
        dataset_test = TypoDataset(os.path.join(args.data_dir, args.test_file), bert_tokenizer, typo_tokenizer)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      batch_size=args.batch_size,
                                                      shuffle=False,
                                                      num_workers=4*args.num_gpus,
                                                      collate_fn=dataset_test.get_collate_fn())

    # BERT encoder (context) and decoder (character) model
    bert_tf_file = os.path.join(args.bert_dir, 'bert_model.ckpt')
    bert_model = CharacterLanguageModel.build_bert_model(bert_config)
    bert_model.load_tf_weights(config=bert_config, tf_checkpoint_path=bert_tf_file)
    char_embeddings_decoder = CharacterLanguageModel.get_char_embeddings_from_bert(
        bert_model.cls.predictions.decoder.weight, bert_tokenizer)
    char_decoder = CharacterLanguageModel.build_char_decoder(char_decoder_config,
                                                             char_embeddings_decoder)

    # The main model
    model = CharacterLanguageModel(args, char_decoder_config, bert_model, char_decoder)

    # Load ckpt if exists
    init_step = 0
    ckpt_manager = CheckPointManager(args.output_dir)
    if args.init_ckpt is not None:
        print(f'Load model ckpt from {args.init_ckpt} (step {args.init_step})')
        model.load_state_dict(torch.load(args.init_ckpt))
        init_step = args.init_step
    else:
        ckpt_status = ckpt_manager.get_latest_checkpoint()
        if ckpt_status:
            init_step, ckpt_fname = ckpt_status
            ckpt_fpath = os.path.join(args.output_dir, ckpt_fname)
            print(f'Load model ckpt from {ckpt_fpath} (step {init_step})')
            ckpt_manager.load_ckpt(model, ckpt_fname)
    global_step = init_step

    if global_step == 0:
        ckpt_manager.save_args(args)

    # DataParallel for num_gpus > 1
    if args.num_gpus > 1:
        model_train = DataParallelModel(model, device_ids=list(range(args.num_gpus)))
    else:
        model_train = model
    model_train.cuda()

    # Train or Test
    val_acc = 0.0
    if args.is_train:
        # Loss function and optimizer
        criteria = CrossEntropyLoss()
        if args.num_gpus > 1:
            criteria = DataParallelCriterion(criteria,
                                             device_ids=list(range(args.num_gpus)))

        if not args.train_bert:
            params = filter(lambda p: p.requires_grad, model.parameters())
        else:
            params = model.get_param_group(finetune_bert=True,
                                           finetune_factor=args.bert_finetune_factor)
        optimizer = fairseq.optim.build_optimizer(args, params)
        optimizer.set_lr = types.MethodType(CharacterLanguageModel.get_set_lr(), optimizer) # Override the set_lr() of the optimizer
        lr_scheduler = fairseq.optim.lr_scheduler.build_lr_scheduler(args, optimizer)
        lr_scheduler.step_update(global_step)

        # Tensorboard log
        writer = SummaryWriter(args.output_dir)

        # Training!!
        model.train()
        train_iter = iter(dataloader_train)
        val_iter = iter(dataloader_val)
        while global_step < args.training_step or global_step == init_step:
            batch_train = next(train_iter)
            start_time = time.time()

            optimizer.zero_grad()
            context_tokens = batch_train['context_tokens'].cuda()
            context_attention_mask = batch_train['context_attention_mask'].cuda()
            correct_token_ids = batch_train['correct_token_ids'].cuda()
            correct_attention_mask = batch_train['correct_attention_mask'].cuda()

            correct_label = correct_token_ids[:, 1:]  # Remove first BOS
            correct_label_mask = correct_attention_mask[:, 1:].float()
            correct_token_ids = correct_token_ids[:, :-1]  # Remove last token that doesn't have label
            correct_attention_mask = correct_attention_mask[:, :-1].float()

            # Model forward
            output_logits = model_train.forward(context_tokens,
                                                context_attention_mask,
                                                correct_token_ids,
                                                correct_attention_mask,
                                                target_correct=correct_label,
                                                return_dict=True)
            loss = criteria(output_logits, targets=correct_label,
                            target_mask=correct_label_mask)

            loss.backward()
            optimizer.step()

            duration = time.time() - start_time

            if global_step % args.display_iter == 0:
                examples_per_sec = args.batch_size / duration
                print_str = f'{datetime.now()}: step {global_step:7}, loss={float(loss.cpu()):.6f}'
                writer.add_scalar('train/loss', float(loss.cpu()), global_step)
                writer.add_scalar('train/learning_rate', optimizer.get_lr(), global_step)
                writer.flush()
                print_str += f', lr={optimizer.get_lr():g} ({examples_per_sec:.1f} it/s, {duration:.3f} s/batch)'
                print(print_str)


            if (global_step % args.eval_iter == 0 and global_step > init_step) or global_step == args.training_step - 1:
                model.eval()

                with torch.no_grad():
                    print(" [Train result]")
                    try:
                        batch_val = next(val_iter)
                    except StopIteration:
                        val_iter = iter(dataloader_val)
                        batch_val = next(val_iter)
                    total_cnt, included_cnt, correct_cnt = 0, 0, 0
                    typo_words, decoder_words, correct_words, decoder_score, lm_scores = \
                            get_batch_result(batch_val, model_train, typo_tokenizer, trie=trie)
                    for j in range(args.batch_size):
                        total_cnt += 1
                        included_cnt += correct_words[j] in decoder_words[j*args.num_beams:(j+1)*args.num_beams]
                        correct_cnt += correct_words[j] == decoder_words[j*args.num_beams]
                        if j < 5:
                            print(f'Typo       : {typo_words[j]}')
                            print(f'Correct    : {correct_words[j]}')
                            print(f'Transformer: {decoder_words[j*args.num_beams:(j+1)*args.num_beams]}')
                            print('')
                    print(f'Total/Included/Correct = {total_cnt} / {included_cnt} / {correct_cnt}')
                    writer.add_scalar('train/accuracy', correct_cnt/total_cnt, global_step)
                    writer.add_scalar('train/included', included_cnt/total_cnt, global_step)
                    writer.flush()

                save_ckpt_fname = ckpt_manager.save_ckpt(model, global_step)
                save_ckpt_fpath = os.path.join(args.output_dir, save_ckpt_fname)
                print(f'Save checkpoint to {save_ckpt_fpath}')

                torch.cuda.empty_cache()

                model.train()

            global_step += 1
            lr_scheduler.step_update(global_step)

    else:  #  args.is_train == False
        model_train.eval()

        # Test!!
        with torch.no_grad():
            results = get_epoch_result(dataloader_test, model_train, typo_tokenizer, trie=trie)
            total_cnt = len(results)
            included_cnt = sum([r['correct'] in r['output'] for r in results])
            correct_cnt = sum([r['correct'] == r['output'][0] for r in results])
            for j in random.sample(range(total_cnt), 5):
                print(f'Typo       : {results[j]["typo"]}')
                print(f'Correct    : {results[j]["correct"]}')
                print(f'Transformer: {results[j]["output"]}')
                print('')
            print(f'Total/Included/Correct = {total_cnt} / {included_cnt} / {correct_cnt}')

            test_name = os.path.basename(args.data_dir).split('_')[-1]
            test_type = args.test_file[:args.test_file.index('.tsv')]
            test_result_dir = os.path.join(args.output_dir, f'test-{global_step}')
            if not os.path.exists(test_result_dir):
                os.makedirs(test_result_dir)
            test_result_fname = f'results_{test_name}_{test_type}_beam{args.num_beams}_ed{args.edit_distance_extra_len}-{args.edit_distance_weight}_lp{args.length_penalty}_bs{args.beam_sort_linear_ed}_fs{args.beam_final_score_normalize_ed}_{"dict" if args.dict_matching else "nodict"}.pkl'
            test_result_path = os.path.join(test_result_dir, test_result_fname)
            print(f'Save val results to {test_result_path}')
            with open(test_result_path, 'wb') as fd:
                pickle.dump(results, fd)

            correct_results = get_epoch_correct_scores(dataloader_test, model, typo_tokenizer)
            test_correct_result_fname = f'correct_{test_name}_{test_type}_beam{args.num_beams}_ed{args.edit_distance_extra_len}-{args.edit_distance_weight}_lp{args.length_penalty}_bs{args.beam_sort_linear_ed}_fs{args.beam_final_score_normalize_ed}_{"dict" if args.dict_matching else "nodict"}.pkl'
            test_correct_result_path = os.path.join(test_result_dir, test_correct_result_fname)
            print(f'Save val results to {test_correct_result_path}')
            with open(test_correct_result_path, 'wb') as fd:
                pickle.dump(correct_results, fd)


    print('done')

def get_words(token_ids, typo_tokenizer):
    """ Print words from token_ids """
    num_words = token_ids.size()[0]
    words = []
    for i in range(num_words):
        tokens = typo_tokenizer.convert_ids_to_tokens(token_ids[i,:])
        while True:
            if tokens[-1] == '<pad>':
                tokens = tokens[:-1]
            else: break
        words.append(''.join(tokens))
    return words

def get_batch_result(batch, model, typo_tokenizer, trie=None):
    context_tokens = batch['context_tokens'].cuda()
    context_attention_mask = batch['context_attention_mask'].cuda()
    typo_token_ids = batch['typo_token_ids'].cuda()
    typo_attention_mask = batch['typo_attention_mask'].cuda()
    correct_token_ids = batch['correct_token_ids'].cuda()
    decoder_token_ids, decoder_scores, lm_scores = model.generate(input_ids=context_tokens,
                                                   attention_mask=context_attention_mask,
                                                   context_attention_mask=context_attention_mask,
                                                   use_cache=False,
                                                   decoder_start_token_id=0,
                                                   bos_token_id=0, pad_token_id=1, eos_token_id=2,
                                                   max_length=64, min_length=3, num_beams=args.num_beams,
                                                   num_return_sequences=args.num_beams,
                                                   typo_token_ids=typo_token_ids,
                                                   length_penalty=args.length_penalty,
                                                   edit_distance_weight=args.edit_distance_weight,
                                                   edit_distance_extra_len=args.edit_distance_extra_len,
                                                   beam_sort_linear_ed=args.beam_sort_linear_ed,
                                                   beam_final_score_normalize_ed=args.beam_final_score_normalize_ed,
                                                   trie=trie)

    typo_words = get_words(typo_token_ids, typo_tokenizer)
    decoder_words = get_words(decoder_token_ids, typo_tokenizer)
    correct_words = get_words(correct_token_ids, typo_tokenizer)
    return typo_words, decoder_words, correct_words, decoder_scores, lm_scores

def get_epoch_result(dataloader, model, typo_tokenizer, trie=None):
    results = []
    for i, batch in enumerate(tqdm(dataloader)):
        typo_words, decoder_words, correct_words, decoder_scores, lm_scores = \
            get_batch_result(batch, model, typo_tokenizer, trie)
        for j in range(len(typo_words)):
            example_id = batch['example_id'][j]
            note_id = batch['note_id'][j]
            typo_word = typo_words[j]
            decoder_words2 = decoder_words[j*args.num_beams:(j+1)*args.num_beams]
            decoder_scores2 = decoder_scores[j*args.num_beams:(j+1)*args.num_beams]
            lm_scores2 = lm_scores[j*args.num_beams:(j+1)*args.num_beams]
            correct_word = correct_words[j]
            results.append({
                'example_id': int(example_id),
                'note_id': int(note_id),
                'typo': typo_word,
                'correct': correct_word,
                'output': decoder_words2,
                'scores': decoder_scores2,
                'lm_scores': lm_scores2,
            })
    return results

def get_epoch_correct_scores(dataloader, model, typo_tokenizer):
    results = []
    for i, batch in enumerate(tqdm(dataloader)):
        context_tokens = batch['context_tokens'].cuda()
        context_attention_mask = batch['context_attention_mask'].cuda()
        typo_token_ids = batch['typo_token_ids'].cuda()
        typo_attention_mask = batch['typo_attention_mask'].cuda()
        correct_token_ids = batch['correct_token_ids'].cuda()
        correct_attention_mask = batch['correct_attention_mask'].cuda()
        typo_words = get_words(typo_token_ids, typo_tokenizer)
        correct_words = get_words(correct_token_ids, typo_tokenizer)

        correct_label = correct_token_ids[:, 1:].unsqueeze(-1)  # Remove first BOS
        correct_label_mask = correct_attention_mask[:, 1:].float()
        correct_token_ids = correct_token_ids[:, :-1]  # Remove last token
        correct_attention_mask = correct_attention_mask[:, :-1]

        # Model forward
        output_logits = model.forward(context_tokens,
                                      context_attention_mask,
                                      correct_token_ids,
                                      correct_attention_mask,
                                      return_dict=True)

        lprobs = torch.nn.functional.log_softmax(output_logits, dim=-1)
        char_scores = lprobs.gather(dim=-1, index=correct_label).squeeze(-1)
        lm_scores = (char_scores * correct_label_mask).sum(-1)
        for j in range(len(typo_words)):
            lm_score = float(lm_scores[j]) / (len(correct_words[j]) - 6) ** args.length_penalty
            ed_score = damerauLevenshtein(typo_words[j][3:-4], correct_words[j][3:-4], similarity=False) \
                * args.edit_distance_weight * -1.0
            if args.beam_sort_linear_ed:
                ed_score *= len(correct_words[j]) - 6
            if args.beam_final_score_normalize_ed:
                ed_score /= (len(correct_words[j]) - 6) ** args.length_penalty
            score = lm_score + ed_score

            example_id = batch['example_id'][j]
            note_id = batch['note_id'][j]
            results.append({
                'example_id': int(example_id),
                'note_id': int(note_id),
                'typo': typo_words[j],
                'correct': correct_words[j],
                'score': score,
                'lm_score': lm_score,
            })

    return results

if __name__ == "__main__":
    main()
