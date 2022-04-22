##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Data Parallel"""
from itertools import chain
import threading
import functools
import torch
from torch.autograd import Variable, Function
import torch.cuda.comm as comm
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper

torch_ver = torch.__version__[:3]

__all__ = ['allreduce', 'DataParallelModel', 'DataParallelCriterion',
           'patch_replication_callback']

def allreduce(*inputs):
    """Cross GPU all reduce autograd operation for calculate mean and
    variance in SyncBN.
    """
    return AllReduce.apply(*inputs)

class AllReduce(Function):
    @staticmethod
    def forward(ctx, num_inputs, *inputs):
        ctx.num_inputs = num_inputs
        ctx.target_gpus = [inputs[i].get_device() for i in range(0, len(inputs), num_inputs)]
        inputs = [inputs[i:i + num_inputs]
                 for i in range(0, len(inputs), num_inputs)]
        # sort before reduce sum
        inputs = sorted(inputs, key=lambda i: i[0].get_device())
        results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
        outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *inputs):
        inputs = [i.data for i in inputs]
        inputs = [inputs[i:i + ctx.num_inputs]
                 for i in range(0, len(inputs), ctx.num_inputs)]
        results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
        outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
        return (None,) + tuple([Variable(t) for tensors in outputs for t in tensors])

class Reduce(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """
    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        return modules

    def gather_generate(self, outputs, output_device):
        ret = []
        for i in range(len(outputs[0])):
            tensors = [output[i] for output in outputs]
            if i == 0:  # Add padding for decoder_token_ids
                max_len = max([t.shape[1] for t in tensors])
                tensors = [torch.cat([t, torch.ones((t.shape[0], max_len - t.shape[1]), dtype=t.dtype, device=t.device)], 1)
                           for t in tensors]
                ret.append(torch._C._gather(tensors, 0, output_device))
            else:  # scores, lm_scores are lists
                ret.append(list(chain(*tensors)))
        return ret

    def generate(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.generate(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = parallel_apply_generate(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
        return self.gather_generate(outputs, self.output_device)  # gather() of this class does not gather, for training

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        kwargs2 = {k: v for k, v in kwargs.items() if k != 'encoder_embeds'}
        inputs, kwargs2 = self.scatter(inputs, kwargs2, self.device_ids)
        if 'encoder_embeds' in kwargs:
            for i, d in enumerate(kwargs2):
                d['encoder_embeds'] = kwargs['encoder_embeds'][i]
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs2[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs2)
        return self.gather(outputs, self.output_device)



class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """
    def forward(self, inputs, *targets, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)

        # new_loss = Reduce.apply(*[o[0] for o in outputs]) / len(outputs)
        # new_loss1 = Reduce.apply(*[o[1][0] for o in outputs]) / len(outputs)
        # new_loss2 = Reduce.apply(*[o[1][1] for o in outputs]) / len(outputs)
        # new_loss3 = Reduce.apply(*[o[1][2] for o in outputs]) / len(outputs)
        # return new_loss, (new_loss1, new_loss2, new_loss3)


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    # assert len(targets) == len(inputs)
    if not targets:
        targets = [None] * len(modules)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
    if torch_ver != "0.3":
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != "0.3":
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(input, **kwargs)
                # output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, target,
                                          kwargs, device),)
                   for i, (module, input, target, kwargs, device) in
                   enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs



def parallel_apply_generate(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module.generate(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def _criterion_parallel_apply2(modules, inputs, inputs2, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    # assert len(targets) == len(inputs)
    if not targets:
        targets = [None] * len(modules)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
    if torch_ver != "0.3":
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, input2, target, kwargs, device=None):
        if torch_ver != "0.3":
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(input, input2, **kwargs)
                # output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, input2, target,
                                          kwargs, device),)
                   for i, (module, input, input2, target, kwargs, device) in
                   enumerate(zip(modules, inputs, inputs2, targets, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], inputs2[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
