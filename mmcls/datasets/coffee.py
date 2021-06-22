import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class coffee(BaseDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py  # noqa: E501
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [
        'CAMERON\'S SPECIALTY COFFEE JAMAICA BLUE MOUNTAIN BLEND POD 12 CT',
        'CAMERON\'S SPECIALTY COFFEE KONA BLEND LIGHT ROAST PODS 12 CT',
        'CAMERON\'S TOASTED SOUTHERN PECAN K CUPS 12 CT',
        'CARIBOU COFFEE KEURIG CARIBOU BLEND MEDIUM ROAST K-CUP 32 CT',
        'CHOCK FULL O\' NUTS MIDTOWN MANHATTAN MEDIUM 20 CT',
        'COMMUNITY COFFEE KEURIG HOT BREAKFAST BLEND MEDIUM ROAST 12 CT',
        'COMMUNITY COFFEE KEURIG HOT CAFE SPECIAL MEDIUM ROAST 12 CT',
        'DUNKIN\' DONUTS DUNKIN DARK K-CUP 10 CT',
        'DUNKIN\' DONUTS KEURIG COFFEE ORIGINAL BLEND K-CUP 32 CT',
        'DUNKIN\' DONUTS KEURIG DUNKIN\' DECAF MEDIUM ROAST COFFEE K-CUP PODS 10 CT',
        'DUNKIN\' DONUTS KEURIG HAZELNUT COFFEE K-CUP PODS 10 CT',
        'DUNKIN\' DONUTS KEURIG HOT 100% COLOMBIAN K-CUP PODS 10 CT',
        'DUNKIN\' DONUTS KEURIG HOT FRENCH VANILLA COFFEE 10 CT',
        'DUNKIN\' DONUTS KEURIG HOT MEDIUM ROAST ORIGINAL BLEND COFFEE 32 CT',
        'DUNKIN\' DONUTS KEURIG ORIGINAL BLEND COFFEE MEDIUM ROAST 10 CT',
        'EIGHT O\'CLOCK KEURIG THE ORIGINAL MEDIUM ROAST PODS 32 CT',
        'FOLGERS CLASSIC ROAST MEDIUM KEURIG K-CUPS 12 CT',
        'FOLGERS KEURIG CARAMEL DRIZZLE K-CUP PODS 12 CT',
        'FOLGERS VANILLA BISCOTTI KEURIG K-CUPS 12 CT',
        'GREEN MOUNTAIN COFFEE BREAKFAST BLEND DECAF LIGHT ROAST K-CUP PODS 32 CT',
        'GREEN MOUNTAIN COFFEE CARAMEL VANILLA CREAM K-CUP PODS 32 CT',
        'GREEN MOUNTAIN COFFEE KEURIG DOUBLE DIAMOND K-CUP 12 CT',
        'GREEN MOUNTAIN COFFEE ROASTERS K-CUP COFFEE DARK ROAST SUMATRA RESERVE 12 CT',
        'GREEN MOUNTAIN COFFEE ROASTERS KEURIG DARK MAGIC DARK ROAST K-CUP 32 CT',
        'GREEN MOUNTAIN COFFEE ROASTERS KEURIG HALF-CAFF MEDIUM ROAST K-CUP PODS VALUE PACK 32 CT',
        'GREEN MOUNTAIN KEURIG COLOMBIA SELECT MEDIUM ROAST K-CUP PODS 12 CT',
        'GREEN MOUNTAIN KEURIG HAZELNUT K-CUP 12 CT',
        'GREEN MOUNTAIN KEURIG HOT NANTUCKET BLEND MEDIUM ROAST COFFEE K-CUPS 12 CT',
        'JAVA HOUSE AUTHENTIC COLD BREW COFFEE COLOMBIAN MEDIUM ROAST LIQUID PODS 6 CT 8.10 OZ',
        'JAVA HOUSE AUTHENTIC COLD BREW COFFEE SUMATRAN DARK ROAST LIQUID PODS 6 CT',
        'KAUAI COFFEE ISLAND SUNRISE MILD ROAST K-CUP 12 CT',
        'KAUAI COFFEE MOCHA MACADAMIA NUT PODS 12 CT',
        'KAUAI COFFEE NA PALI COAST DARK ROAST K-CUP 12 CT',
        'KAUAI COFFEE VANILLA MACADAMIA NUT K-CUPS 12 CT',
        'KEURIG HOT GEVALIA KAFFE K-CUP COFFEE DARK ROYAL ROAST 12 CT',
        'KEURIG LAVAZZA GRAN AROMA MEDIUM ROAST 10 CT',
        'KEURIG LAVAZZA GRAN SELEZIONE DARK ROAST 10 CT',
        'KRISPY KREME DOUGHNUTS KEURIG CLASSIC DECAF K-CUP 12 CT',
        'KRISPY KREME DOUGHNUTS KEURIG CLASSIC MEDIUM ROAST COFFEE K-CUP 32 CT',
        'LAVAZZA KEURIG CLASSICO COFFEE 10 CT',
        'LAVAZZA KEURIG PERFETTO EXPRESSO ROAST COFFEE 10 CT',
        'MAXWELL HOUSE K-CUP COFFEE ORIGINAL ROAST MEDIUM 12 CT',
        'MAXWELL HOUSE KEURIG BREWED HOUSE BLEND DECAF MEDIUM 12 CT',
        'MAXWELL HOUSE KEURIG HOT BREAKFAST BLEND LIGHT 12 CT',
        'MAXWELL HOUSE KEURIG HOUSE BLEND MEDIUM 12 CT',
        'MAXWELL HOUSE KEURIG ORIGINAL ROAST MEDIUM K-CUP 32 CT',
        'MCCAFE KEURIG PREMIUM ROAST MEDIUM PODS 32 CT',
        'MELITTA SUPER PREMIUM 4 COFFEE FILTERS 100 CT',
        'MELITTA SUPER PREMIUM COFFEE FILTERS 2 NATURAL BROWN CONE 100 CT',
        'PEET\'S COFFEE FRENCH DEEP ROAST CUPS 10 CT',
        'PEET\'S COFFEE K-CUP COFFEE DARK ROAST FRENCH ROAST 32 CT',
        'PEET\'S COFFEE KEURIG ALMA DE LA TIERRA DARK ROAST COFFEE 10 CT',
        'PEET\'S COFFEE KEURIG BIG BANG MEDIUM ROAST COFFEE K-CUPS 10 CT',
        'PEET\'S COFFEE KEURIG CAFE DOMINGO MEDIUM ROAST COFFEE 10 CT',
        'PEET\'S COFFEE KEURIG COLOMBIA LUMINOSA LIGHT ROAST COFFEE 10 CT',
        'PEET\'S COFFEE KEURIG COSTA RICA AURORA LIGHT ROAST COFFEE 10 CT',
        'PEET\'S COFFEE KEURIG DECAF ESPECIAL MEDIUM ROAST COFFEE 10 CT',
        'PEET\'S COFFEE KEURIG FARL FRENCH ROAST K-CUP 10 CT',
        'PEET\'S COFFEE KEURIG HOUSE BLEND DARK ROAST COFFEE 10 CT',
        'PEET\'S COFFEE KEURIG HOUSE BLEND DECAF DARK ROAST COFFEE K-CUPS 10 CT',
        'PEET\'S COFFEE KEURIG MAJOR DICKASON\'S BLEND DARK ROAST 32 CT',
        'PEET\'S COFFEE KEURIG MAJOR DICKASON\'S BLEND DARK ROAST COFFEE 32 CT',
        'PEET\'S COFFEE KEURIG MAJOR DICKASON\'S BLEND K-CUP 10 CT',
        'PEET\'S COFFEE KEURIG NICARAGUA ADELANTE MEDIUM ROAST COFFEE 10 CT',
        'PRIVATE SELECTION COFFEE KONA BLEND MEDIUM K-CUP 48 CT',
        'PRIVATE SELECTION COLOMBIAN SWISS DECAF K-CUP 12 CT',
        'PRIVATE SELECTION KEURIG BREWED GUATEMALAN DARK COFFEE 48 CT',
        'PRIVATE SELECTION SIGNATURE BLEND MEDIUM K-CUPS 12 CT',
        'SEATTLE\'S BEST COFFEE KEURIG HOT BREAKFAST BLEND MEDIUM ROAST 10 CT',
        'SEATTLE\'S BEST COFFEE KEURIG HOUSE BLEND MEDIUM ROAST 10 CT',
        'SF BAY COFFEE FOG CHASER ONE CUPS 30 CT',
        'SF BAY COFFEE FRENCH ROAST ONE CUPS 30 CT',
        'SIMPLE TRUTH ORGANIC DARK ROAST COFFEE PODS 12 CT',
        'STARBUCKS BLONDE SUNRISE BLEND K-CUP 10 CT',
        'STARBUCKS FRESH BREW GROUND COFFEE CANS BREAKFAST BLEND 8 CT',
        'STARBUCKS KEURIG BLONDE ROAST WITH 2X CAFFEINE 10 CT',
        'STARBUCKS KEURIG BREAKFAST BLEND PODS 32 CT',
        'STARBUCKS KEURIG CAFFE VERONA DARK ROAST K-CUP 32 CT',
        'STARBUCKS KEURIG COLOMBIA SINGLE-ORIGIN MEDIUM ROAST COFFEE K-CUP PODS 10 CT',
        'STARBUCKS KEURIG DARK ROAST WITH 2X CAFFEINE 10 CT',
        'STARBUCKS KEURIG FRENCH ROAST PODS 32 CT',
        'STARBUCKS KEURIG HOT BREAKFAST BLEND MEDIUM ROAST GROUND COFFEE 10 CT',
        'STARBUCKS KEURIG HOT BRIGHT SKY BLEND BLONDE ROAST 10 CT',
        'STARBUCKS KEURIG HOT CAFFE VERONA DARK ROAST GROUND COFFEE K-CUP PODS 10 CT',
        'STARBUCKS KEURIG HOT CARAMEL FLAVORED COFFEE 10 CT',
        'STARBUCKS KEURIG HOT FRENCH DARK ROAST 10 CT',
        'STARBUCKS KEURIG HOT HOUSE BLEND MEDIUM ROAST 10 CT',
        'STARBUCKS KEURIG HOT ITALIAN ROAST DARK ROAST GROUND COFFEE K-CUP',
        'STARBUCKS KEURIG HOT SUMATRA SINGLE ORIGIN COFFEE DARK ROAST K-CUPS 10 CT',
        'STARBUCKS KEURIG HOT TOASTED GRAHAM FLAVORED GROUND COFFEE SIGNATURE COLLECTION 10 CT',
        'STARBUCKS KEURIG HOT VANILLA FLAVORED COFFEE 10 CT',
        'STARBUCKS KEURIG MEDIUM ROAST COFFEE WITH ESSENTIAL VITAMINS 10 CT',
        'STARBUCKS KEURIG MEDIUM ROAST COFFEE WITH GOLDEN TURMERIC 10 CT',
        'STARBUCKS KEURIG MEDIUM ROAST WITH 2X CAFFEINE 10 CT',
        'STARBUCKS KEURIG PIKE PLACE MEDIUM ROAST COFFEE K-CUP PODS 32 CT',
        'STARBUCKS KEURIG PIKE PLACE ROAST MEDIUM ROAST K-CUPS 10 CT',
        'STARBUCKS KEURIG PUMPKIN SPICE K-CUP 32 CT',
        'STARBUCKS KEURIG SUMATRA SINGLE-ORIGIN DARK ROAST COFFEE K-CUP PODS 32 CT',
        'STARBUCKS KEURIG VERANDA BLEND BLONDE ROAST K-CUP 32 CT'
    ]

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
