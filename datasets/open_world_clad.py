import copy
import os
import itertools
import torch
import torchvision

from typing import Optional, List, Sequence, Callable, Dict, Any
from collections import defaultdict
from torchvision import transforms as T
from PIL import Image

from util.clad_utils import load_obj_img_dic, create_domain_dicts


BASE_CLAD_CLASS_NAMES = ["Car", "Truck", "Tram", "Cyclist", "Tricycle", "Pedestrian"] # categori_id : [0, 1, 2, 3, 4, 5]
UNK_CLASS = ["unknown"]

CLAD_CLASS_NAMES = {}

T1_CLASS_NAMES = ["Car", "Truck", "Tram"]

T2_CLASS_NAMES = ["Cyclist", "Tricycle"]

T3_CLASS_NAMES = ["Pedestrian"]

CLAD_CLASS_NAMES['CLAD'] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, UNK_CLASS))


print(CLAD_CLASS_NAMES)

class OWCladDetection(torch.utils.data.Dataset):
    """
    A class that creates a Clad-D dataset, which will covers a given domain. Class incremental style datasets
    isn't supported in this dataset.

    :param root: root path of the folder where files are stored
    :param ids: Ids of the images that should be in the dataset
    :param transform: Transform to be applied to images before returning
    :param meta: Any string with usefull meta information.
    """

    def __init__(self, 
                 args,
                 root: str,
                 image_set: str,
                 annot_file: str,
                 transform: Optional[Callable] = None,
                 meta: str = None,
                 ):
        super(OWCladDetection).__init__()

        split = annot_file.split('_')[-1].split('.')[0]

        self.args = args
        self.image_set = image_set
        self.img_folder = os.path.join(root, 'SSLAD-2D', 'labeled', split)
        self.ids = self.extract_clad_fns(root, image_set)
        self.transform = transform if transform is not None else get_transform(split == 'train')
        self.meta = meta
        self.CLASS_NAMES = CLAD_CLASS_NAMES['CLAD']

        self.obj_annotations, self.img_annotations = load_obj_img_dic(annot_file)
        self._remove_empty_images()
        self.img_anns = self._create_index()

    def extract_clad_fns(self, root, image_set):
        splits_dir = os.path.join(root, 'SSLAD-2D', 'labeled')
        splits_dir = os.path.join(splits_dir, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [int(x.strip()) for x in f.readlines()]
        return file_names

    def _remove_empty_images(self):
        """
        Required because torchvision models can't handle empty lists for bbox in targets
        """
        non_empty_images = set()
        for obj in self.obj_annotations.values():
            non_empty_images.add(obj["image_id"])
        self.ids = [img_id for img_id in self.ids if img_id in non_empty_images]

    def _create_index(self):
        img_anns = defaultdict(list)
        for ann in self.obj_annotations.values():
            img_anns[ann['image_id']].append(ann)
        return img_anns

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        total_num_class = self.args.num_classes # 7
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in  copy.copy(entry):
            if annotation["category_id"] not in known_classes:
                annotation["category_id"] = total_num_class - 1
        return entry

    @property
    def targets(self):
        """
        Get a list of all category ids, required for Avalanche.
        """
        targets = []
        for img_id in self.img_anns:
            targets.extend(obj['category_id'] for obj in self.img_anns[img_id])
        return torch.tensor(targets)

    def _load_target(self, index: str):
        img_id = self.ids[index]
        img_objects = self.img_anns[img_id]
        sizes = self.img_annotations[img_id]['width'], self.img_annotations[img_id]['height']

        # # Pre-processing following OWDetection
        if 'train' in self.image_set:
            img_objects = self.remove_prev_class_and_unk_instances(img_objects)
        elif 'test' in self.image_set or 'val' in self.image_set:
            img_objects = self.label_known_class_and_unknown(img_objects)
        elif 'ft' in self.image_set:
            img_objects = self.remove_unknown_instances(img_objects)

        boxes = []
        for obj in img_objects:
            bbox = obj["bbox"]
            # Convert from x, y, h, w to x0, y0, x1, y1
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([obj["category_id"] for obj in img_objects], dtype=torch.int64)
        area = torch.as_tensor([obj["area"] for obj in img_objects])
        iscrowd = torch.as_tensor([obj["iscrowd"] for obj in img_objects], dtype=torch.int64)

        # Targets should all be tensors
        target = {"boxes": boxes, "labels": labels, "image_id": torch.as_tensor(img_id, dtype=torch.int64),
                  "sizes": torch.as_tensor(sizes, dtype=torch.int64), "area": area, "iscrowd": iscrowd}

        return target

    def _load_image(self, index):
        file_name = self.img_annotations[self.ids[index]]['file_name']
        return Image.open(os.path.join(self.img_folder, file_name)).convert('RGB')

    def __getitem__(self, index):
        image = self._load_image(index)
        instances = self._load_target(index) # boxes, labels, image_id, sizes(width, height), area, iscrowd

        if self.transform is not None:
            image, instances = self.transform(image, instances)

        w, h = instances["sizes"]
        target = dict(
            image_id=instances['image_id'],
            labels=instances['labels'],
            area=instances['area'],
            boxes=instances['boxes'],
            orig_size=instances['sizes'],
            size=instances['sizes'],
            iscrowd=instances['iscrowd']
        )

        return image, target

    def __len__(self):
        return len(self.ids)


def get_matching_detection_set(root: str, annot_file: str, match_fn: Callable, transform=None,
                               meta: str = None) -> OWCladDetection:
    """
    Creates OWCladDetection set from a match_fn

    :param root: root path of where to look for pickled object files
    :param annot_file: annotation file from root
    :param match_fn: A function that takes a sample, the obj and img dicts and return T/F if a sample should be
                     in the dataset
    :param transform: Transformation to apply to images. If None, _default_transform will be applied.
    :param meta: optional meta information in a str that will be stored with the dataset.
    :return: OWCladDetection object
    """

    _, img_dic = load_obj_img_dic(annot_file)
    img_ids = [image for image in img_dic if match_fn(image, img_dic)]

    return OWCladDetection(root, img_ids, annot_file, transform, meta)


def get_cladd_domain_sets(root: str, annot_file: str, domains: Sequence[str], transform: Callable = None,
                          match_fn: Callable = None) -> Sequence[OWCladDetection]:
    """
    :param root: Root directory of the dataset
    :param annot_file: the annotation file for the dataset
    :param domains: the domains that should be included in the domains sets (e.g. ['period', 'city'])
    :param match_fn: a method that returns True if a given sample should be in the dataset.
    :param transform
    """

    if match_fn is None:
        def match_fn(*args): return True

    domain_dicts = create_domain_dicts(domains)
    domain_set = []
    for domain_dict in domain_dicts:
        domain_match_fn = create_match_dict_fn_img(domain_dict)
        ds = get_matching_detection_set(root, annot_file, lambda *args: domain_match_fn(*args) and match_fn(*args),
                                        transform, meta='-'.join(domain_dict.values()))
        domain_set.append(ds)
    return domain_set


def create_match_dict_fn_img(match_dict: Dict[Any, Any]) -> Callable:
    """
    Creates a method that returns true if the image specified by the img_id
    is in the specified domain of the given match_dict.
    :param match_dict: dictionary that should match the objects
    :return: a function that evaluates to true if the object is from the given date
    """

    def match_fn(img_id: int, img_dic: Dict[str, Dict]) -> bool:
        img_annot = img_dic[img_id]
        for key, value in match_dict.items():
            if isinstance(value, List):
                if img_annot[key] not in value:
                    return False
            else:
                if img_annot[key] != value:
                    return False
        else:
            return True

    return match_fn


def create_val_from_trainset(trainset: OWCladDetection, root, val_transform, split, proportion=0.1):
    cut_off = int((1.0 - proportion) * len(trainset))
    all_imgs = trainset.ids
    trainset.ids = all_imgs[:cut_off]
    val_ids = all_imgs[cut_off:]
    return OWCladDetection(root, val_ids, os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations',
                                                     f'instance_{split}.json'), val_transform, trainset.meta)


# Below adapted from pytorch vision example on detection, but removed unnecessary code.

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.transform = T.RandomHorizontalFlip(prob)

    def __call__(self, image, target):
        self.transform(image)
        return image, target
    
    # def __call__(self, image, target):
    #     if random.random() < self.prob:
    #         height, width = image.shape[-2:]
    #         image = image.flip(-1)
    #         bbox = target["boxes"]
    #         bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
    #         target["boxes"] = bbox
    #     return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


def get_transform(train):
    transform_arr = [ToTensor()]
    if train:
        transform_arr.append(RandomHorizontalFlip(0.5))
    return Compose(transform_arr)