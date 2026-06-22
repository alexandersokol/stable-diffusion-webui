import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from collections import defaultdict
from random import shuffle, choices

import random
import tqdm
from modules import devices, shared, images
import re

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

re_numbers_at_start = re.compile(r"^[-\d]+\s*")


class DatasetEntry:
    def __init__(self, filename=None, filename_text=None, latent_dist=None, latent_sample=None, cond=None, cond_text=None, pixel_values=None, weight=None, cache_file=None):
        self.filename = filename
        self.filename_text = filename_text
        self.weight = weight
        self.latent_dist = latent_dist
        self.latent_sample = latent_sample
        self.cond = cond
        self.cond_text = cond_text
        self.pixel_values = pixel_values
        self.cache_file = cache_file

    def copy_metadata(self):
        return DatasetEntry(
            filename=self.filename,
            filename_text=self.filename_text,
            cond_text=self.cond_text,
            pixel_values=self.pixel_values,
            cache_file=self.cache_file,
        )


class PersonalizedBase(Dataset):
    def __init__(self, data_root, width, height, repeats, flip_p=0.5, placeholder_token="*", model=None, cond_model=None, device=None, template_file=None, include_cond=False, batch_size=1, gradient_step=1, shuffle_tags=False, tag_drop_out=0, latent_sampling_method='once', varsize=False, use_weight=False, cache_mode=None, cache_dir=None):
        re_word = re.compile(shared.opts.dataset_filename_word_regex) if shared.opts.dataset_filename_word_regex else None

        self.placeholder_token = placeholder_token
        self.cache_mode = cache_mode if cache_mode is not None else getattr(shared.opts, "training_dataset_latent_cache_mode", "memory")
        if self.cache_mode not in ("memory", "disk"):
            raise ValueError(f"unknown textual inversion dataset cache mode: {self.cache_mode}")

        self.cache_dir = None
        self.owns_cache_dir = False
        if self.cache_mode == "disk":
            if cache_dir is None:
                self.cache_dir = Path(tempfile.mkdtemp(prefix="textual-inversion-latents-"))
                self.owns_cache_dir = True
            else:
                self.cache_dir = Path(cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.dataset = []

        with open(template_file, "r") as file:
            lines = [x.strip() for x in file.readlines()]

        self.lines = lines

        assert data_root, 'dataset directory not specified'
        assert os.path.isdir(data_root), "Dataset directory doesn't exist"
        assert os.listdir(data_root), "Dataset directory is empty"

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]

        self.shuffle_tags = shuffle_tags
        self.tag_drop_out = tag_drop_out
        groups = defaultdict(list)

        print("Preparing dataset...")
        for path in tqdm.tqdm(self.image_paths):
            alpha_channel = None
            if shared.state.interrupted:
                raise Exception("interrupted")
            try:
                image = images.read(path)
                #Currently does not work for single color transparency
                #We would need to read image.info['transparency'] for that
                if use_weight and 'A' in image.getbands():
                    alpha_channel = image.getchannel('A')
                image = image.convert('RGB')
                if not varsize:
                    image = image.resize((width, height), PIL.Image.BICUBIC)
            except Exception:
                continue

            text_filename = f"{os.path.splitext(path)[0]}.txt"
            filename = os.path.basename(path)

            if os.path.exists(text_filename):
                with open(text_filename, "r", encoding="utf8") as file:
                    filename_text = file.read()
            else:
                filename_text = os.path.splitext(filename)[0]
                filename_text = re.sub(re_numbers_at_start, '', filename_text)
                if re_word:
                    tokens = re_word.findall(filename_text)
                    filename_text = (shared.opts.dataset_filename_join_string or "").join(tokens)

            npimage = np.array(image).astype(np.uint8)
            npimage = (npimage / 127.5 - 1.0).astype(np.float32)

            torchdata = torch.from_numpy(npimage).permute(2, 0, 1).to(device=device, dtype=torch.float32)
            latent_sample = None

            with devices.autocast():
                latent_dist = model.encode_first_stage(torchdata.unsqueeze(dim=0))

            #Perform latent sampling, even for random sampling.
            #We need the sample dimensions for the weights
            if latent_sampling_method == "deterministic":
                if isinstance(latent_dist, DiagonalGaussianDistribution):
                    # Works only for DiagonalGaussianDistribution
                    latent_dist.std = 0
                else:
                    latent_sampling_method = "once"
            latent_sample = model.get_first_stage_encoding(latent_dist).squeeze().to(devices.cpu)

            if use_weight and alpha_channel is not None:
                channels, *latent_size = latent_sample.shape
                weight_img = alpha_channel.resize(latent_size)
                npweight = np.array(weight_img).astype(np.float32)
                #Repeat for every channel in the latent sample
                weight = torch.from_numpy(np.broadcast_to(npweight, [channels] + latent_size).copy())
                #Normalize the weight to a minimum of 0 and a mean of 1, that way the loss will be comparable to default.
                weight -= weight.min()
                weight /= weight.mean()
            elif use_weight:
                #If an image does not have a alpha channel, add a ones weight map anyway so we can stack it later
                weight = torch.ones(latent_sample.shape)
            else:
                weight = None

            if latent_sampling_method == "random":
                entry = DatasetEntry(filename=path, filename_text=filename_text, latent_dist=latent_dist, weight=weight)
            else:
                entry = DatasetEntry(filename=path, filename_text=filename_text, latent_sample=latent_sample, weight=weight)

            if not (self.tag_drop_out != 0 or self.shuffle_tags):
                entry.cond_text = self.create_text(filename_text)

            if include_cond and not (self.tag_drop_out != 0 or self.shuffle_tags):
                with devices.autocast():
                    entry.cond = cond_model([entry.cond_text]).to(devices.cpu).squeeze(0)
            self.cache_entry_tensors(entry, len(self.dataset))
            groups[image.size].append(len(self.dataset))
            self.dataset.append(entry)
            del torchdata
            del latent_dist
            del latent_sample
            del weight

        self.length = len(self.dataset)
        self.groups = list(groups.values())
        assert self.length > 0, "No images have been found in the dataset."
        self.batch_size = min(batch_size, self.length)
        self.gradient_step = min(gradient_step, self.length // self.batch_size)
        self.latent_sampling_method = latent_sampling_method

        if len(groups) > 1:
            print("Buckets:")
            for (w, h), ids in sorted(groups.items(), key=lambda x: x[0]):
                print(f"  {w}x{h}: {len(ids)}")
            print()

    def cache_entry_tensors(self, entry, index):
        if self.cache_mode != "disk":
            return

        payload = {}
        for name in ("latent_sample", "latent_dist", "cond", "weight"):
            value = getattr(entry, name)
            if value is not None:
                payload[name] = value
                setattr(entry, name, None)

        entry.cache_file = self.cache_dir / f"{index:08d}.pt"
        torch.save(payload, entry.cache_file)

    def load_cached_tensors(self, entry):
        if self.cache_mode != "disk" or entry.cache_file is None:
            return entry

        try:
            payload = torch.load(entry.cache_file, map_location=devices.cpu, weights_only=False)
        except TypeError:
            payload = torch.load(entry.cache_file, map_location=devices.cpu)
        for name, value in payload.items():
            setattr(entry, name, value)

        return entry

    def create_text(self, filename_text):
        text = random.choice(self.lines)
        tags = filename_text.split(',')
        if self.tag_drop_out != 0:
            tags = [t for t in tags if random.random() > self.tag_drop_out]
        if self.shuffle_tags:
            random.shuffle(tags)
        text = text.replace("[filewords]", ','.join(tags))
        text = text.replace("[name]", self.placeholder_token)
        return text

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        entry = self.dataset[i]
        if self.cache_mode == "disk":
            entry = self.load_cached_tensors(entry.copy_metadata())
        if self.tag_drop_out != 0 or self.shuffle_tags:
            entry.cond_text = self.create_text(entry.filename_text)
        if self.latent_sampling_method == "random":
            entry.latent_sample = shared.sd_model.get_first_stage_encoding(entry.latent_dist).to(devices.cpu)
            if self.cache_mode == "disk":
                entry.latent_dist = None
        return entry

    def __del__(self):
        if getattr(self, "owns_cache_dir", False) and getattr(self, "cache_dir", None) is not None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)


class GroupedBatchSampler(Sampler):
    def __init__(self, data_source: PersonalizedBase, batch_size: int):
        super().__init__(data_source)

        n = len(data_source)
        self.groups = data_source.groups
        self.len = n_batch = n // batch_size
        expected = [len(g) / n * n_batch * batch_size for g in data_source.groups]
        self.base = [int(e) // batch_size for e in expected]
        self.n_rand_batches = nrb = n_batch - sum(self.base)
        self.probs = [e%batch_size/nrb/batch_size if nrb>0 else 0 for e in expected]
        self.batch_size = batch_size

    def __len__(self):
        return self.len

    def __iter__(self):
        b = self.batch_size

        for g in self.groups:
            shuffle(g)

        batches = []
        for g in self.groups:
            batches.extend(g[i*b:(i+1)*b] for i in range(len(g) // b))
        for _ in range(self.n_rand_batches):
            rand_group = choices(self.groups, self.probs)[0]
            batches.append(choices(rand_group, k=b))

        shuffle(batches)

        yield from batches


class PersonalizedDataLoader(DataLoader):
    def __init__(self, dataset, latent_sampling_method="once", batch_size=1, pin_memory=False):
        super(PersonalizedDataLoader, self).__init__(dataset, batch_sampler=GroupedBatchSampler(dataset, batch_size), pin_memory=pin_memory)
        if latent_sampling_method == "random":
            self.collate_fn = collate_wrapper_random
        else:
            self.collate_fn = collate_wrapper


class BatchLoader:
    def __init__(self, data):
        self.cond_text = [entry.cond_text for entry in data]
        self.cond = [entry.cond for entry in data]
        self.latent_sample = torch.stack([entry.latent_sample for entry in data]).squeeze(1)
        if all(entry.weight is not None for entry in data):
            self.weight = torch.stack([entry.weight for entry in data]).squeeze(1)
        else:
            self.weight = None
        #self.emb_index = [entry.emb_index for entry in data]
        #print(self.latent_sample.device)

    def pin_memory(self):
        self.latent_sample = self.latent_sample.pin_memory()
        return self

def collate_wrapper(batch):
    return BatchLoader(batch)

class BatchLoaderRandom(BatchLoader):
    def __init__(self, data):
        super().__init__(data)

    def pin_memory(self):
        return self

def collate_wrapper_random(batch):
    return BatchLoaderRandom(batch)
