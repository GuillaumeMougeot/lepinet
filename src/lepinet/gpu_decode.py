"""nvJPEG GPU-decode reader: CPU workers read raw JPEG bytes, the GPU decodes.

The default fastai pipeline decodes JPEGs on CPU worker processes, which costs ~1.2 GB of host
(anon) memory per worker -- so a high `num_workers` (needed to feed a fast GPU over a slow mount)
blows up host memory and OOMs on a tight cgroup. Moving decode to the GPU makes the CPU side pure
I/O: a worker holds one ~80 KB JPEG, not a decode+resize+collate pipeline, cutting per-worker
memory ~10x and removing the worker-count/OOM bind entirely.

Measured trade-off (journal/2026-07-ucloud-throughput.md): for a *small* model (effnetv2s) this
is a MEMORY win, not a speed win -- the model is the throughput ceiling (~1100 img/s) either way,
and even a B200 can't exceed it. It pays off in speed only on a heavier, compute-bound model. And
it must be used correctly or it is *slower* than CPU decode:

  1. OVERLAP decode with compute (`overlapped_batches`): decode batch N+1 on a side CUDA stream
     while the model runs batch N. Without this, decode+model run serially and throughput ~halves.
  2. gc every N batches, not every batch (`gc_collect_every`): mini_trainer's cosine head leaves a
     GPU reference cycle that traps each batch's backward graph; a per-batch gc.collect breaks it
     but halves throughput. Every ~8 batches keeps GPU memory flat with negligible cost.
  3. Bigger batch on a big GPU: the model is the ceiling, so batch size is the real speed lever.

Correctness: `decode_jpeg` throws on the occasional CMYK/corrupt/non-JPEG file where PIL coped;
those are decoded one-by-one and any straggler replaced with a grey frame rather than killing the
batch. Grayscale images are expanded to 3 channels. Resize + normalise happen on the GPU; wire
your augmentation after (fastai already runs aug_transforms on the GPU as batch_tfms).
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JpegBytesDataset(Dataset):
    """Returns (raw_jpeg_bytes_as_uint8_tensor, label_tensor). CPU workers do I/O only -- no
    decode -- so per-worker memory is ~one JPEG, not a decode pipeline. `paths` are relative to
    `img_dir`; `labels` is an (N, n_levels) int64 array/tensor."""

    def __init__(self, paths, labels, img_dir):
        self.paths = paths
        self.labels = labels
        self.img_dir = str(img_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(os.path.join(self.img_dir, str(self.paths[i])), "rb") as f:
            buf = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
        return buf, torch.as_tensor(self.labels[i])


def collate_jpeg_bytes(batch):
    """Keep the variable-length byte tensors as a list (they can't be stacked); stack labels."""
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


def decode_batch(byte_list, size, device, *, augment=None,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """bytes -> decoded, resized, (optionally augmented,) normalised float batch on `device`.

    `augment(x)` is an optional GPU-tensor-in/GPU-tensor-out callable applied before normalise
    (e.g. flips/affine, or fastai batch_tfms). `size` is the square model-input side.
    """
    try:
        imgs = decode_jpeg(byte_list, device=device, mode=ImageReadMode.RGB)
    except RuntimeError:
        imgs = []
        for b in byte_list:
            try:
                imgs.append(decode_jpeg(b, device=device, mode=ImageReadMode.RGB))
            except RuntimeError:
                imgs.append(torch.full((3, size, size), 114, dtype=torch.uint8, device=device))

    out = torch.empty((len(imgs), 3, size, size), device=device)
    for j, im in enumerate(imgs):
        if im.shape[0] == 1:
            im = im.expand(3, -1, -1)
        out[j] = F.interpolate(im.unsqueeze(0).float(), (size, size),
                               mode="bilinear", align_corners=False).squeeze(0)
    out /= 255.0
    if augment is not None:
        out = augment(out)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (out - m) / s


def overlapped_batches(dl: DataLoader, size, device, *, augment=None,
                       prefetch=3, **decode_kwargs) -> Iterator:
    """Yield (decoded_gpu_batch, labels), double-buffered: a producer thread decodes the next
    batch on a side CUDA stream while the consumer trains on the current one, so decode overlaps
    model compute (~1.7x over serial in profiling). `dl` must yield (byte_list, labels), i.e. use
    `collate_fn=collate_jpeg_bytes`.

    Without overlap, iterate the loader directly and call `decode_batch` inline -- simpler, but
    decode and model run serially.
    """
    stream = torch.cuda.Stream()
    q: queue.Queue = queue.Queue(maxsize=prefetch)

    def producer():
        for byte_list, yb in dl:
            with torch.cuda.stream(stream):
                xb = decode_batch(byte_list, size, device, augment=augment, **decode_kwargs)
            q.put((xb, yb.to(device, non_blocking=True), stream.record_event()))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        xb, yb, ev = item
        torch.cuda.current_stream().wait_event(ev)
        yield xb, yb


def make_jpeg_bytes_loader(paths, labels, img_dir, batch_size, *, num_workers,
                           sampler=None, shuffle=None, prefetch_factor=6):
    """A DataLoader over JpegBytesDataset with the bytes collate. Pair its output with
    `overlapped_batches(...)` (or `decode_batch` inline) to decode on the GPU."""
    if shuffle is None:
        shuffle = sampler is None
    return DataLoader(
        JpegBytesDataset(paths, labels, img_dir),
        batch_size=batch_size, num_workers=num_workers, sampler=sampler,
        shuffle=shuffle, collate_fn=collate_jpeg_bytes,
        persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
