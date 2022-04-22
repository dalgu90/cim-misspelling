#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os
import json
import torch


class CheckPointManager(object):
    def __init__(self, checkpoint_dir, max_to_keep=5, info_fname='ckpt-info'):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.info_fname = info_fname

    def save_ckpt_info(self, info):
        with open(os.path.join(self.checkpoint_dir, self.info_fname), 'w') as fd:
            for train_iter in sorted(info.keys()):
                fd.write(f'{train_iter}\t{info[train_iter]}\n')

    def load_ckpt_info(self):
        ckpt_info = {}
        info_fpath = os.path.join(self.checkpoint_dir, self.info_fname)
        if os.path.exists(info_fpath):
            with open(info_fpath, 'r') as fd:
                for l in fd:
                    split = l.strip().split("\t")
                    train_iter, ckpt_fname = int(split[0]), split[1]
                    ckpt_info[train_iter] = ckpt_fname
        return ckpt_info

    def clean_up_ckpt_info(self, info):
        info_ret = {}
        for train_iter, ckpt_fname in info.items():
            ckpt_fpath = os.path.join(self.checkpoint_dir, ckpt_fname)
            if os.path.exists(ckpt_fpath):
                info_ret[train_iter] = ckpt_fname
        return info_ret

    def get_latest_checkpoint(self):
        info = self.load_ckpt_info()
        info = self.clean_up_ckpt_info(info)
        if info:
            train_iter = max(info.keys())
            return train_iter, info[train_iter]
        return None

    def save_ckpt(self, model, train_iter, ckpt_fname=None):
        # Load ckpt info
        info = self.load_ckpt_info()
        info = self.clean_up_ckpt_info(info)

        # Save new ckpt
        if not ckpt_fname:
            ckpt_fname = f'ckpt-{train_iter}.pkl'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        ckpt_fpath = os.path.join(self.checkpoint_dir, ckpt_fname)
        torch.save(model.state_dict(), ckpt_fpath)

        # Add ckpt info while removing duplicate
        if ckpt_fname in info.values():
            for train_iter2, ckpt_fname2 in info.items():
                if ckpt_fname2 == ckpt_fname:
                    del info[train_iter2]
                    break
        info[train_iter] = ckpt_fname

        # Delete old ckpts
        num_ckpt_del = max(0, len(info) - self.max_to_keep)
        train_iters_del = sorted(info.keys())[:num_ckpt_del]
        for train_iter2 in train_iters_del:
            os.remove(os.path.join(self.checkpoint_dir, info[train_iter2]))
            del info[train_iter2]

        # Save ckpt info
        self.save_ckpt_info(info)
        return ckpt_fname

    def load_ckpt(self, model, ckpt_fname):
        ckpt_fpath = os.path.join(self.checkpoint_dir, ckpt_fname)
        model.load_state_dict(torch.load(ckpt_fpath))

    def save_args(self, args):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        with open(os.path.join(self.checkpoint_dir, 'args.json'), 'w') as fd:
            json.dump(vars(args), fd)
