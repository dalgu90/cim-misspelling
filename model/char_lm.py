#!/usr/bin/env python
# -*- coding=utf-8 -*-

#
# Character Langauge Model of CIM
#
# Internally, it uses the `transformers` implementation of BERT and BART
#


import os
import json
import copy
import argparse
import numpy as np
import random
import fairseq
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.generation_utils import top_k_top_p_filtering
from fastDamerauLevenshtein import damerauLevenshtein
import multiprocessing

DEFAULT_MAX_CHARACTER_POSITIONS = 64

# char_tokens = list("0123456789abcdefghijklmnopqrstuvwxyz+-*/^.,;:=!?'()[]{}")
char_tokens = list("0123456789abcdefghijklmnopqrstuvwxyz+-*/^.,;:=!?'()[]{}&")
special_tokens_fairseq = ['<s>', '<pad>', '</s>', '<unk>']
special_tokens_bert = ['[CLS]', '[PAD]', '[SEP]', '[UNK]']  # CLS and SEP don't exactly match


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, sum_logprobs2=0.0):
        """
        Add a new hypothesis to the list.
        second score is for auxiliary use
        """
        # len - 1: ignore the first bos token
        score = sum_logprobs / len(hyp) ** self.length_penalty
        score2 = sum_logprobs2 / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, score2))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class CharacterLanguageModel(transformers.PretrainedBartModel):
    def __init__(self,
                 args,
                 config,
                 bert_model,
                 char_decoder):
        super().__init__(config)

        self.args = args
        self.config = config
        self.final_logits_bias = torch.nn.parameter.Parameter(
            torch.zeros(1, self.config.vocab_size))
        self.encoder = bert_model.bert
        self.decoder = char_decoder
        self.train_bert = args.train_bert
        if not self.train_bert:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.bos, self.pad, self.eos, self.unk = special_tokens_fairseq
        self.ed_pool_master = multiprocessing.pool.ThreadPool(processes=1)
        self.ed_pool_worker = multiprocessing.Pool(processes=4)

    @classmethod
    def get_bert_config(cls, bert_config_file):
        with open(bert_config_file, 'r') as fd:
            config_orig = json.load(fd)

        bert_config = transformers.BertConfig(
            vocab_size=config_orig['vocab_size'],
            hidden_size=config_orig['hidden_size'],
            num_hidden_layers=config_orig['num_hidden_layers'],
            num_attention_heads=config_orig['num_attention_heads'],
            intermediate_size=config_orig['intermediate_size'],
            hidden_act=config_orig['hidden_act'],
            hidden_dropout_prob=config_orig['hidden_dropout_prob'],
            attention_probs_dropout_prob=config_orig['attention_probs_dropout_prob'],
            max_position_embeddings=config_orig['max_position_embeddings'],
            type_vocab_size=config_orig['type_vocab_size'],
            initializer_range=config_orig['initializer_range'],
            layer_norm_eps=1e-12  # This is not on the original config json file
        )
        return bert_config

    @classmethod
    def build_bert_model(cls, bert_config):
        bert_model = transformers.BertForPreTraining(bert_config)
        return bert_model

    @classmethod
    def get_bert_tokenizer(cls, bert_vocab_file):
        bert_tokenizer = transformers.BertTokenizer(
            bert_vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True
        )
        return bert_tokenizer

    @classmethod
    def get_char_embeddings_from_bert(cls, bert_embeddings, bert_tokenizer):
        # Get char embeddings of BERT, in favor of `fairseq.data.Dictionary`
        char_word_ids = []
        char_word_ids.append(bert_tokenizer.cls_token_id)  # CLS -> BOS
        char_word_ids.append(bert_tokenizer.pad_token_id)
        char_word_ids.append(bert_tokenizer.sep_token_id)  # SEP -> EOS
        char_word_ids.append(bert_tokenizer.unk_token_id)
        char_word_ids.extend(bert_tokenizer.convert_tokens_to_ids(char_tokens))
        if isinstance(bert_embeddings, torch.nn.modules.sparse.Embedding):
            embedding_matrix = bert_embeddings(torch.tensor(char_word_ids)).detach()
        elif isinstance(bert_embeddings, torch.Tensor):
            embedding_matrix = bert_embeddings[char_word_ids, :].detach()
        char_embeddings = torch.nn.Embedding.from_pretrained(embedding_matrix,
                                freeze=False, padding_idx=1)
        return char_embeddings

    @classmethod
    def get_char_decoder_config(cls, bert_config_file, args):
        """Generate config for Character Decoder using BERT config."""
        with open(bert_config_file, 'r') as fd:
            config_orig = json.load(fd)

        bart_config = transformers.BartConfig(
            is_decoder=True,
            vocab_size=len(char_tokens) + len(special_tokens_fairseq),
            d_model=config_orig['hidden_size'],
            decoder_layers=args.decoder_layers,
            # decoder_attention_heads=config_orig['num_attention_heads'],
            # decoder_ffn_dim=config_orig['intermediate_size'],
            # activation_function=config_orig['hidden_act'],
            # dropout=config_orig['hidden_dropout_prob'],
            # attention_dropout=config_orig['attention_probs_dropout_prob'],
            max_position_embeddings=DEFAULT_MAX_CHARACTER_POSITIONS,
            init_str=config_orig['initializer_range']
        )
        return bart_config

    @classmethod
    def build_char_decoder(cls, bart_config, char_embedding):
        """Create BART decoder for character LM."""
        bart_decoder = transformers.modeling_bart.BartDecoder(bart_config,
                                                              char_embedding)
        return bart_decoder

    def forward(self,
                input_ids_context,  # input_ids
                attention_mask_context=None,  # attention_mask
                input_ids_correct=None,  # decoder_input_ids
                attention_mask_correct=None,  # decoder_attention_mask
                target_correct=None,  # Used by L-softmax, only for training
                encoder_outputs=None,
                encoder_embeds=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        # From Huggingface's implementation of Bart

        # encoder outputs consists of (sequence_output, pooled_output, all_hidden, all_attn)
        if encoder_embeds is not None:
            encoder_outputs = (encoder_embeds,)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids_context,
                attention_mask=attention_mask_context,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Remove EOS from decoder inputs and make masks correspondingly
        decoder_input_ids = input_ids_correct - input_ids_correct.eq(2).int()  # make EOS(2) to PAD(1)
        if attention_mask_correct is not None:
            decoder_padding_mask = attention_mask_correct.eq(0) | input_ids_correct.eq(2)
        else:
            decoder_padding_mask = decoder_input_ids.eq(1)
        _, _, causal_mask = transformers.modeling_bart._prepare_bart_decoder_inputs(
            self.config,
            input_ids=input_ids_correct,
            decoder_input_ids=None,
            decoder_padding_mask=attention_mask_correct,
            causal_mask_dtype=self.decoder.embed_tokens.weight.dtype,
        )
        # decoder_input_ids = decoder_input_ids[:, :-1]
        # decoder_padding_mask = decoder_padding_mask[:, :-1]
        # causal_mask = causal_mask[:-1, :-1]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask_context,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_output_logits = F.linear(decoder_outputs[0], self.decoder.embed_tokens.weight,
                                    bias=self.final_logits_bias)

        output_logits = transformers.modeling_outputs.Seq2SeqLMOutput(
            logits=lm_output_logits,
        ).logits

        if not kwargs.get('return_encoder_embeds'):
            return output_logits
        else:
            return encoder_outputs[0], output_logits

        # return lm_output_logits

        # if not return_dict:
            # return decoder_outputs + encoder_outputs

        # return transformers.modeling_outputs.Seq2SeqModelOutput(
            # last_hidden_state=decoder_outputs.last_hidden_state,
            # past_key_values=decoder_outputs.past_key_values,
            # decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        # )

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return transformers.modeling_bart._make_linear_from_emb(
                self.decoder.embed_tokens)  # make it on the fly

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        return {
            "input_ids_context": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "input_ids_correct": decoder_input_ids,
            "attention_mask_context": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")



    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs_logits = self(**model_inputs, return_dict=True)
            next_token_logits = outputs_logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            # if "past_key_values" in outputs:
                # past = outputs.past_key_values
            # elif "mems" in outputs:
                # past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids, []


    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example with beam search."""

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        # Typo and token appended
        edit_distance_weight = model_kwargs['edit_distance_weight']
        if edit_distance_weight != 0.0:
            edit_distance_extra_len =  model_kwargs['edit_distance_extra_len']
            # edit_distance = [[0 for _ in range(vocab_size)] for _ in range(batch_size * num_beams)]
            typo_token_ids = [token_ids.tolist() for token_ids in model_kwargs['typo_token_ids']]
            typo_token_ids_strip = [l[:l.index(1)] for l in typo_token_ids]

        while cur_len < max_length:
            # Compute edit distance of each beam and typo
            input_ids_list = None
            if edit_distance_weight != 0.0:
                input_ids_list = input_ids.tolist()
                ed_async_result = self.ed_pool_master.apply_async(
                        self._compute_edit_distances,
                        (input_ids_list, typo_token_ids_strip, vocab_size, edit_distance_extra_len)
                )

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )
            outputs_logits = self(**model_inputs, return_dict=True)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs_logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            # if "past_key_values" in outputs:
                # past = outputs.past_key_values
            # elif "mems" in outputs:
                # past = outputs.mems

            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            scores = self.postprocess_next_token_scores(
                scores=scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            if model_kwargs['trie']:
                scores = self.postprocess_next_token_scores_dict(
                    scores=scores,
                    input_ids=input_ids,
                    trie=model_kwargs['trie']
                )

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            # Compute edit distance of each beam and typo
            if edit_distance_weight != 0.0:
                edit_distance_mat = ed_async_result.get()
                edit_distance_tensor = torch.Tensor(edit_distance_mat).view(batch_size * num_beams, vocab_size).to(input_ids.device)
            else:
                edit_distance_tensor = torch.zeros((batch_size * num_beams, vocab_size), dtype=torch.float, device=input_ids.device)

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                next_scores2 = next_scores - edit_distance_weight * edit_distance_tensor
                next_scores3 = next_scores - (cur_len ** length_penalty) * edit_distance_weight * edit_distance_tensor

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)
                next_scores2 = next_scores2.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)
                next_scores3 = next_scores3.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Choose top-k using MAP score, but keep only log-likelihood of decoder
                if model_kwargs['beam_sort_linear_ed'] == True:
                    _, next_tokens = torch.topk(next_scores3, 2 * num_beams, dim=1, largest=True, sorted=True)
                else:
                    _, next_tokens = torch.topk(next_scores2, 2 * num_beams, dim=1, largest=True, sorted=True)
                next_scores = torch.gather(next_scores, 1, next_tokens)
                next_scores2 = torch.gather(next_scores2, 1, next_tokens)
                next_scores3 = torch.gather(next_scores3, 1, next_tokens)

                if model_kwargs['beam_final_score_normalize_ed'] == True:
                    next_scores_final = next_scores2
                else:
                    next_scores_final = next_scores3

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score, beam_token_score_final) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_scores_final[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(),
                            beam_token_score_final.item(),
                            beam_token_score.item()
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id, beam_token_score_final))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores_final[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            beam_scores_final = beam_scores.new([x[3] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            ## Print intermediate results
            if 'print_beam_steps' in model_kwargs and model_kwargs['print_beam_steps']:
                chars = special_tokens_fairseq + char_tokens
                def get_word(token_ids):
                    return ''.join([chars[t] for t in token_ids])
                print_batch_idx = 0
                print(f'Beam search intermediate {cur_len}')
                for ii in range(num_beams):
                    eff_idx = print_batch_idx * num_beams + ii
                    print(f'{get_word(input_ids[eff_idx])}: {beam_scores[eff_idx]:-9.6f} / {beam_scores_final[eff_idx]:-9.6f}')
                for score_, token_ids_, score2_ in sorted(generated_hyps[print_batch_idx].beams, key=lambda x: x[0], reverse=True):
                    print(f'({get_word(token_ids_)}): {score_:-9.6f} / {score2_:-9.6f}')

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )


        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx],
                    beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores_final[effective_beam_id].item()
                final_score2 = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score, final_score2)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        final_scores = []
        lm_scores = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()
                sent_lengths[effective_batch_idx] = len(best_hyp[1])
                best.append(best_hyp[1])
                final_scores.append(best_hyp[0])
                lm_scores.append(best_hyp[2])
                # effective_batch_idx = output_num_return_sequences_per_batch * i + j
                # best_hyp = sorted_hyps.pop()[1]
                # sent_lengths[effective_batch_idx] = len(best_hyp)
                # best.append(best_hyp)

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id

        return decoded, final_scores, lm_scores


    def postprocess_next_token_scores_dict(self, scores, input_ids, trie):
        if not trie:
            return scores
        bsz, vocab_size = scores.shape
        constraint = [[float('-inf') for _ in range(vocab_size)] for _ in range(bsz)]

        for i, prefix_ids in enumerate(input_ids.tolist()):
            cand_ids = trie.get_candidate_chars(prefix_ids[1:]) # ignore first bos token
            for j in cand_ids:
                constraint[i][j] = 0.0
        scores += torch.Tensor(constraint).to(scores.device)

        return scores


    def parameters(self):
        """ The _parameters of replicas are empty, so use _former_parameters """
        if len(list(super().parameters())):
            ret =  super().parameters()
        elif hasattr(self, "_former_parameters"):
            ret = (p for m in self.modules() for p in m._former_parameters.values())
        elif hasattr(self, "_parameters"):
            ret = (p for m in self.modules() for p in m._parameters.values())
        return ret


    def _compute_edit_distances(self, input_ids_list, typo_token_ids_strip, vocab_size, edit_distance_extra_len):
        """
        Input:
            input_ids_list: list of list of int list (batch_size, num_beams, cur_len)
            typo_token_ids_strip: int list (batch_size)
            vocab_size: int
            edit_distance_extra_len: int
        Output:
            edit_distance_tensor: list of list of float list (batch_size, num_bean, vocab_size)
        """
        batch_size = len(typo_token_ids_strip)
        num_beams = len(input_ids_list) // batch_size
        cur_len = len(input_ids_list[0])

        dataset = [
                (input_ids_list[i*num_beams:(i+1)*num_beams], typo_token_ids_strip[i], vocab_size, edit_distance_extra_len)
                for i in range(batch_size)
        ]
        num_process = min(batch_size, 4)
        edit_distance_mat = list(self.ed_pool_worker.imap(_edit_distance_pool_job, dataset, chunksize=len(dataset)//num_process))

        return edit_distance_mat


    def get_param_group(self, finetune_bert=False, finetune_factor=0.1):
        if not finetune_bert:
            param_group = [{'params': list(self.parameters()), 'lr_factor': 1.0}]
        else:
            bert_params = list(self.encoder.parameters())
            bert_params_set = set(bert_params)
            other_params = [p for p in list(self.parameters()) if p not in bert_params_set]
            param_group = [{'params': other_params, 'lr_factor': 1.0},
               {'params': bert_params, 'lr_factor': finetune_factor}]
        return param_group


    @staticmethod
    def get_set_lr():
        def set_lr(self, lr):
            for param_group in self.optimizer.param_groups:
                if 'lr_factor' in param_group:
                    param_group['lr'] = lr * param_group['lr_factor']
                else:
                    param_group['lr'] = lr
        return set_lr



def _edit_distance_pool_job(data):
    input_ids_ex, typo_token_ids_strip_ex, vocab_size, edit_distance_extra_len = data
    num_beams, cur_len = len(input_ids_ex), len(input_ids_ex[0])
    ed_submat = [[0.0 for _ in range(vocab_size)] for _ in range(num_beams)]
    typo_token_ids_temp = typo_token_ids_strip_ex[:cur_len+1+edit_distance_extra_len]
    for idx1 in range(num_beams):
        temp = input_ids_ex[idx1] + [0]
        for idx2 in range(vocab_size):
            temp[-1] = idx2
            ed_submat[idx1][idx2] = damerauLevenshtein(temp, typo_token_ids_temp, similarity=False)
    return ed_submat


# def _trie_score_pool_job(data):
    # input_ids_ex, trie, vocab_size = data
def _trie_score_pool_job(input_ids_ex, trie, vocab_size):
    num_examples, cur_len = len(input_ids_ex), len(input_ids_ex[0])
    score = [[float('-inf') for _ in range(vocab_size)] for _ in range(num_examples)]

    for i, prefix_ids in enumerate(input_ids_ex):
        cand_ids = trie.get_candidate_chars(prefix_ids[1:]) # ignore first bos token
        for j in cand_ids:
            score[i][j] = 0.0
    return score


class CharTokenizer(object):
    def __init__(self, max_length=DEFAULT_MAX_CHARACTER_POSITIONS):
        self.max_length = max_length
        self.bos, self.pad, self.eos, self.unk = special_tokens_fairseq
        self.bos_index, self.pad_index, self.eos_index, self.unk_index = 0, 1, 2, 3
        self.char_to_id = {}
        self.id_to_char = {}
        for i, c in enumerate(special_tokens_fairseq + char_tokens):
            self.char_to_id[c] = i
            self.id_to_char[i] = c


    def tokenize(self, typo, eos_bos=True, padding_end=False, max_length=None,
                 output_token_ids=False):
        assert isinstance(typo, str)

        max_seq_len = self.max_length - 2 if eos_bos else self.max_length

        tokens = []
        attention_mask = []
        for c in typo[:max_seq_len]:
            if c in self.char_to_id:
                tokens.append(c)
            else:
                tokens.append(self.unk)
            attention_mask.append(1)

        if eos_bos:
            tokens.insert(0, '<s>')  # self.char_dict.bos_word not exist
            tokens.append(self.eos)
            attention_mask.insert(0, 1)
            attention_mask.append(1)
        if padding_end:
            max_length = max_length if max_length is not None else self.max_length
            while len(tokens) < max_length:
                tokens.append(self.pad)
                attention_mask.append(0)

        if output_token_ids:
            return self.convert_tokens_to_ids(tokens), attention_mask
        else:
            return tokens, attention_mask

    def convert_tokens_to_ids(self, tokens):
        return [self.char_to_id[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().detach().tolist()
        return [self.id_to_char[i] for i in ids]


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SmoothCrossEntropyLoss, self).__init__()

    def forward(self, logits, target, smoothing=0.0):
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(-1)
        lprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
        smooth_loss = -lprobs.sum(dim=-1)
        eps_i = smoothing / lprobs.size(-1)
        loss = (1.0 - smoothing) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets=None, target_mask=None):
        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(-1)
        lprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -lprobs.gather(dim=-1, index=targets).squeeze(-1)
        if target_mask is not None:
            loss = ((nll_loss * target_mask).sum(-1) / target_mask.sum(-1)).mean()
        else:
            loss = nll_loss.mean()
        return loss


class Trie(object):
    def __init__(self, char_tokenizer, eos_token_id=2):
        self.char_tokenizer = char_tokenizer
        self.eos_token_id = eos_token_id
        self._trie = {}

    def add_word_ids(self, word_token_ids):
        trie = self._trie
        for token_id in word_token_ids:
            if token_id not in trie:
                trie[token_id] = {}
            trie = trie[token_id]
        trie[self.eos_token_id] = {}

    def add_words(self, words):
        for word in words:
            try:
                word_token_ids = self.char_tokenizer.convert_tokens_to_ids(word)
                self.add_word_ids(word_token_ids)
            except KeyError as e:
                continue

    def get_candidate_chars(self, prefix_ids):
        trie = self._trie
        for token_id in prefix_ids:
            if token_id not in trie:
                return []
            trie = trie[token_id]
        return trie.keys()
