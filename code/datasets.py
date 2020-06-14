from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import six
import string
import sys
import torch


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = self.load_bbox()
        split_dir = os.path.join(data_dir, 'embeddings_' + split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        bbox = self.bbox[key]
        data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        wrong_bbox = self.bbox[wrong_key]
        wrong_img_name = '%s/images/%s.jpg' % \
            (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, wrong_imgs, embedding, key  # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        bbox = self.bbox[key]
        data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)

class CUBDatasetLazy(torch.utils.data.Dataset):
    """CUB dataset.

    Class for CUB Dataset with precomputed embeddings. Reads
    images constantly with PIL and doesn't load them at once.
    To load and keep them as an attribute, use CUBDatasetEager.
    If training dataset, mismatching image is also returned.

    Attributes:
        embeddings(torch.Tensor): embeddings of captions.
        image_filenames(list): filename of image corresponding
            to each embedding (at the same index).
        class_ids(list): class of image and embeddings (at the
            same index).
        dataset_dir(str): directory of data.
        image_dir(str): directory of actual images relative to
            dataset_dir.
        train(bool): whether this is the training dataset.
        synthetic_ids(dict): correspondence between
            real and synthetic IDs. Meant for use
            during testing.
        bboxes(dict): keys are the filenames
            of images and values the bounding box to
            retain a proper image to body ratio.
        transform(torchvision Transform): transform applied to every PIL image
            (as is read from image_dir).
    """

    def __init__(self, dataset_dir, image_dir, embedding_dir,
                 available_classes=None, train=None):
        """Init.

        Args:
            dataset_dir(str): root directory of dataset.
            image_dir(str): image directory w.r.t. dataset_dir.
            embedding_dir(str): embedding directory w.r.t
                dataset_dir.
            available_classes(str, optional): txt file to define
                restrict the classes used from the predefined
                train/test split, default=`None`.
            train(bool, optional): indicating whether training
                on this dataset. If not provided, it is determined
                by the embedding_dir name.
        """

        #########################################
        ########## parse pickle files ###########
        #########################################

        embeddings_fn = os.path.join(dataset_dir, embedding_dir,
                                     'char-CNN-RNN-embeddings.pickle')
        with open(embeddings_fn, 'rb') as emb_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            embeddings = torch.tensor(pickle.load(emb_fp, encoding='latin1'))  # pylint: disable=not-callable

        class_ids_fn = os.path.join(dataset_dir, embedding_dir,
                                    'class_info.pickle')
        with open(class_ids_fn, 'rb') as cls_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            class_ids = pickle.load(cls_fp, encoding='latin1')

        img_fns_fn = os.path.join(dataset_dir, embedding_dir,
                                  'filenames.pickle')
        with open(img_fns_fn, 'rb') as fns_fp:
            # latin1 enc hack bcs pickle compatibility issue between python2 and 3
            img_fns = pickle.load(fns_fp, encoding='latin1')

        ####################################################
        ####################################################

        if available_classes:  # if available_classes is set
                               # keep only them
            # get class ids used in dataset
            with open(os.path.join(dataset_dir, available_classes), 'r') as avcls:
                available_class_ids = [int(line.strip().split('.')[0])
                                       for line in avcls.readlines()]

            idcs = [i for i, cls_id in enumerate(class_ids) if cls_id in available_class_ids]

            self.embeddings = embeddings[idcs]
            self.image_filenames = [img_fns[i] for i in idcs]
            self.class_ids = [cls_id for cls_id in class_ids if cls_id in available_class_ids]

        else:  # if available_classes is not set, keep them all
            self.embeddings = embeddings
            self.image_filenames = img_fns
            self.class_ids = class_ids

        unique_ids = set(self.class_ids)
        self.synthetic_ids = dict(zip(unique_ids, range(len(unique_ids))))

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir

        if train is not None:
            self.train = train
        else:
            # if train is not set, `embedding_dir` should be embeddings_{train, test}
            self.train = (embedding_dir.split('_')[1] == 'train')

        self.bboxes = _load_bboxes(dataset_dir)

        # crop to bbox, make 3 channels if grayscale
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: _bbox_crop(*x)),
            transforms.Lambda(lambda x: transforms.Grayscale(3)(x) if _is_grayscale(x) else x),
            transforms.Resize(304),
            transforms.RandomRotation(5),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        """Return len of dataset.

        Returns:
            Number of images in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Returns an image, its embedding, and maybe
        a mismatching image.

        Retrieve an image, one of its embeddings and,
        if this is a training dataset, a mismatching image.
        Class ID is last returned value.

        Arguments:
            idx(int): index.

        Returns:
            An image as a torch.Tensor of size (3,256,256),
            if training a mismatching image and one of its
            embeddings, and its class id.
        """

        image_fn = self.image_filenames[idx]
        image = Image.open(os.path.join(self.dataset_dir, self.image_dir, image_fn + '.jpg'))

        if not self.train:
            return (self.transform((image, self.bboxes[image_fn])),
                    self.synthetic_ids[self.class_ids[idx]])

        rand_caption = torch.randint(10, (1,)).item()
        embedding = self.embeddings[idx, rand_caption]

        while True:
            # get an image from a different class (match-aware discr)
            mis_idx = torch.randint(len(self), (1,)).item()
            if self.class_ids[idx] != self.class_ids[mis_idx]:
                break

        mis_image_fn = self.image_filenames[mis_idx]
        mis_image = Image.open(os.path.join(self.dataset_dir, self.image_dir,
                                            mis_image_fn + '.jpg'))

        return (self.transform((image, self.bboxes[image_fn])),
                self.transform((mis_image, self.bboxes[mis_image_fn])),
                embedding, self.synthetic_ids[self.class_ids[idx]])

    def embeddings_by_class(self):
        """Fetches the embeddings per class.

        Yields:
            torch.Tensor with embeddings of size
            (#, 10, 1024) and the corresponding
            int synthetic ID.
        """

        prev = 0

        while True:
            curr_id = self.class_ids[prev]

            for curr in range(prev + 1, len(self)):
                if self.class_ids[curr] != curr_id:
                    break  # break at first point where id changes

            if curr == prev:  # handle case with one instance in class
                yield self.embeddings[prev][None, ...], self.synthetic_ids[curr_id]
            else:
                yield self.embeddings[prev:curr], self.synthetic_ids[curr_id]

            prev = curr

            if curr == len(self) - 1:
                break

def _load_bboxes(dataset_dir):
    """Retrieve bounding boxes.

    Builds a dictionary of {filename: bounding_box} pairs
    to crop images to 75% body to image ratio.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        A dictionary of image filename: list of bounding
        box coordinates key-value pairs.
    """

    # id 4xcoords
    df_bboxes = pd.read_csv(os.path.join(dataset_dir, 'bounding_boxes.txt'),
                            delim_whitespace=True, header=None).astype(int)
    # id fn
    df_corr_fns = pd.read_csv(os.path.join(dataset_dir, 'images.txt'),
                              delim_whitespace=True, header=None)

    bbox_dict = {
        os.path.splitext(df_corr_fns.iloc[i][1])[0]: df_bboxes.iloc[i][1:].tolist()
        for i in range(len(df_bboxes))
    }

    return bbox_dict

def _bbox_crop(image, bbox):
    """Crop PIL.Image according to bbox.

    Args:
        image(PIL.Image): image to crop
        bbox(iterable): iterable with 4 elements.

    Returns:
        Cropped image.
    """

    width, height = image.size
    ratio = int(max(bbox[2], bbox[3]) * 0.75)
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    y_low = max(0, center_y - ratio)
    y_high = min(height, center_y + ratio)
    x_low = max(0, center_x - ratio)
    x_high = min(width, center_x + ratio)
    image = image.crop([x_low, y_low, x_high, y_high])

    return image

def _is_grayscale(image):
    """Return if image is grayscale.

    Assert if image only has 1 channel.

    Args:
        image(PIL.Image): image to check.

    Returns:
        bool indicating whether image is grayscale.
    """

    try:
        # channel==1 is 2nd channel
        image.getchannel(1)
        return False
    except ValueError:
        return True

class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset for synthetic samples.

    Dataset to store and retrieve synthetic samples rather than
    holding them all in RAM. Only cares for the samples it
    was used to store. Stores with sequential filenames akin
    to indices for trivial indexing.

    Attributes:
        n_sample(int): number of samples.
        sample_key(str): key of dictionary used to store
            and retrieve samples.
        label_key(str): key of dictionary used to store
            and retrieve corresponding labels.
        template_fn(str): formattable filename for each
            different sample.
    """

    def __init__(self, dataset_dir=None):
        """Init.

        Args:
            dataset_dir(str, optional): directory of dataset,
                default=directory 'dataset' under the
                invisible-to-git cache directory specified
                in configuration file.
        """

        if dataset_dir is None:
            dataset_dir = os.path.join(CACHE_DIR, 'dataset')

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        self.n_sample = 0
        self.sample_key = 'sample'
        self.label_key = 'label'
        self.fn_template = os.path.join(dataset_dir, 'sample_{}.pt')

    @classmethod
    def existing(cls, dataset_dir=None):
        """Init from existing directory.

        Args:
            dataset_dir(str, optional): directory of dataset,
                default=directory 'dataset' under the
                invisible-to-git cache directory specified
                in configuration file.
        """

        obj = cls(dataset_dir)
        obj.n_sample = len(os.listdir(dataset_dir))
        return obj

    def _getitem(self, idx):
        """__getitem__ but only for ints.

        Args:
            idx(int): index.

        Returns:
            torch.Tensors sample and label.
        """

        if idx < - len(self) or idx >= len(self):
            raise IndexError('Index {} out of range'.format(idx))

        if idx < 0:
            idx += len(self)

        # saved as cpu tensors
        sample_dict = torch.load(self.fn_template.format(idx))
        return sample_dict[self.sample_key], sample_dict[self.label_key]

    def __getitem__(self, idx):
        """Loads and returns a sample and its label.

        Args:
            idx(int|slice|torch.Tensor|list): index/indices
                of sample(s).

        Returns:
            torch.Tensors sample(s) and label(s).
        """

        if torch.is_tensor(idx):
            if idx.ndim == 0:
                idx = idx.item()
            else:
                idx = list(idx.numpy())

        if isinstance(idx, int):
            return self._getitem(idx)

        if isinstance(idx, slice):
            # slice (for kNN etc)
            samples, labels = [], []
            for i in range(*idx.indices(len(self))):
                sample, label = self._getitem(i)
                samples.append(sample)
                labels.append(label)

            if not samples:
                raise IndexError('No elements corresponding to {}'.format(idx))

            return torch.stack(samples), torch.stack(labels)

        if isinstance(idx, list):
            samples, labels = [], []
            for i in idx:
                sample, label = self._getitem(i)
                samples.append(sample)
                labels.append(label)

            if not samples:
                raise IndexError('No elements corresponding to {}'.format(idx))

            return torch.stack(samples), torch.stack(labels)

        raise IndexError('Unhandled index type')

    def __len__(self):
        """Returns number of stored samples."""
        return self.n_sample

    def save_pairs(self, samples, label):
        """Saves sample-label pairs.

        Saves pairs of samples and their corresponding label
        (assumed to be the same for all samples, thus only an
        integer is expected) with a filename specified by the
        template and order of receival.

        Args:
            samples(torch.tensor): batch of samples.
            label(int): their corresponding label.
        """

        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)  # pylint: disable=not-callable

        samples = samples.cpu()
        label = label.cpu()

        sample_dict = {self.label_key: label}

        for i in range(samples.size(0)):
            sample_dict[self.sample_key] = samples[i]
            torch.save(sample_dict, self.fn_template.format(self.n_sample))
            self.n_sample += 1
