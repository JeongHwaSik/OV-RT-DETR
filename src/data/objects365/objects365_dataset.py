"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask

from src.core import register

__all__ = ['Objects365Detection']


@register
class Objects365Detection(torchvision.datasets.CocoDetection):
    """
    Objects365_v1 has COCO format
    """

    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        super(Objects365Detection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = super(Objects365Detection, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img.size[::-1])  # spatial_size is (H, W) for torchvision

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # add text for open-world detection
        target['text'] = list(o365_category2name.values()) # TODO: differences b/t obj365v1_class_texts.json

        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # (x, y, w, h) -> (x, y, x, y)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [o365_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]

        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


o365_category2name = {1: 'person', 2: 'sneakers', 3: 'chair', 4: 'hat', 5: 'lamp', 6: 'bottle', 7: 'cabinet/shelf', 8: 'cup', 9: 'car', 10: 'glasses', 11: 'picture/frame', 12: 'desk', 13: 'handbag', 14: 'street lights', 15: 'book', 16: 'plate', 17: 'helmet', 18: 'leather shoes', 19: 'pillow', 20: 'glove', 21: 'potted plant', 22: 'bracelet', 23: 'flower', 24: 'tv', 25: 'storage box', 26: 'vase', 27: 'bench', 28: 'wine glass', 29: 'boots', 30: 'bowl', 31: 'dining table', 32: 'umbrella', 33: 'boat', 34: 'flag', 35: 'speaker', 36: 'trash bin/can', 37: 'stool', 38: 'backpack', 39: 'couch', 40: 'belt', 41: 'carpet', 42: 'basket', 43: 'towel/napkin', 44: 'slippers', 45: 'barrel/bucket', 46: 'coffee table', 47: 'suv', 48: 'toy', 49: 'tie', 50: 'bed', 51: 'traffic light', 52: 'pen/pencil', 53: 'microphone', 54: 'sandals', 55: 'canned', 56: 'necklace', 57: 'mirror', 58: 'faucet', 59: 'bicycle', 60: 'bread', 61: 'high heels', 62: 'ring', 63: 'van', 64: 'watch', 65: 'sink', 66: 'horse', 67: 'fish', 68: 'apple', 69: 'camera', 70: 'candle', 71: 'teddy bear', 72: 'cake', 73: 'motorcycle', 74: 'wild bird', 75: 'laptop', 76: 'knife', 77: 'traffic sign', 78: 'cell phone', 79: 'paddle', 80: 'truck', 81: 'cow', 82: 'power outlet', 83: 'clock', 84: 'drum', 85: 'fork', 86: 'bus', 87: 'hanger', 88: 'nightstand', 89: 'pot/pan', 90: 'sheep', 91: 'guitar', 92: 'traffic cone', 93: 'tea pot', 94: 'keyboard', 95: 'tripod', 96: 'hockey', 97: 'fan', 98: 'dog', 99: 'spoon', 100: 'blackboard/whiteboard', 101: 'balloon', 102: 'air conditioner', 103: 'cymbal', 104: 'mouse', 105: 'telephone', 106: 'pickup truck', 107: 'orange', 108: 'banana', 109: 'airplane', 110: 'luggage', 111: 'skis', 112: 'soccer', 113: 'trolley', 114: 'oven', 115: 'remote', 116: 'baseball glove', 117: 'paper towel', 118: 'refrigerator', 119: 'train', 120: 'tomato', 121: 'machinery vehicle', 122: 'tent', 123: 'shampoo/shower gel', 124: 'head phone', 125: 'lantern', 126: 'donut', 127: 'cleaning products', 128: 'sailboat', 129: 'tangerine', 130: 'pizza', 131: 'kite', 132: 'computer box', 133: 'elephant', 134: 'toiletries', 135: 'gas stove', 136: 'broccoli', 137: 'toilet', 138: 'stroller', 139: 'shovel', 140: 'baseball bat', 141: 'microwave', 142: 'skateboard', 143: 'surfboard', 144: 'surveillance camera', 145: 'gun', 146: 'life saver', 147: 'cat', 148: 'lemon', 149: 'liquid soap', 150: 'zebra', 151: 'duck', 152: 'sports car', 153: 'giraffe', 154: 'pumpkin', 155: 'piano', 156: 'stop sign', 157: 'radiator', 158: 'converter', 159: 'tissue ', 160: 'carrot', 161: 'washing machine', 162: 'vent', 163: 'cookies', 164: 'cutting/chopping board', 165: 'tennis racket', 166: 'candy', 167: 'skating and skiing shoes', 168: 'scissors', 169: 'folder', 170: 'baseball', 171: 'strawberry', 172: 'bow tie', 173: 'pigeon', 174: 'pepper', 175: 'coffee machine', 176: 'bathtub', 177: 'snowboard', 178: 'suitcase', 179: 'grapes', 180: 'ladder', 181: 'pear', 182: 'american football', 183: 'basketball', 184: 'potato', 185: 'paint brush', 186: 'printer', 187: 'billiards', 188: 'fire hydrant', 189: 'goose', 190: 'projector', 191: 'sausage', 192: 'fire extinguisher', 193: 'extension cord', 194: 'facial mask', 195: 'tennis ball', 196: 'chopsticks', 197: 'electronic stove and gas stove', 198: 'pie', 199: 'frisbee', 200: 'kettle', 201: 'hamburger', 202: 'golf club', 203: 'cucumber', 204: 'clutch', 205: 'blender', 206: 'tong', 207: 'slide', 208: 'hot dog', 209: 'toothbrush', 210: 'facial cleanser', 211: 'mango', 212: 'deer', 213: 'egg', 214: 'violin', 215: 'marker', 216: 'ship', 217: 'chicken', 218: 'onion', 219: 'ice cream', 220: 'tape', 221: 'wheelchair', 222: 'plum', 223: 'bar soap', 224: 'scale', 225: 'watermelon', 226: 'cabbage', 227: 'router/modem', 228: 'golf ball', 229: 'pine apple', 230: 'crane', 231: 'fire truck', 232: 'peach', 233: 'cello', 234: 'notepaper', 235: 'tricycle', 236: 'toaster', 237: 'helicopter', 238: 'green beans', 239: 'brush', 240: 'carriage', 241: 'cigar', 242: 'earphone', 243: 'penguin', 244: 'hurdle', 245: 'swing', 246: 'radio', 247: 'CD', 248: 'parking meter', 249: 'swan', 250: 'garlic', 251: 'french fries', 252: 'horn', 253: 'avocado', 254: 'saxophone', 255: 'trumpet', 256: 'sandwich', 257: 'cue', 258: 'kiwi fruit', 259: 'bear', 260: 'fishing rod', 261: 'cherry', 262: 'tablet', 263: 'green vegetables', 264: 'nuts', 265: 'corn', 266: 'key', 267: 'screwdriver', 268: 'globe', 269: 'broom', 270: 'pliers', 271: 'volleyball', 272: 'hammer', 273: 'eggplant', 274: 'trophy', 275: 'dates', 276: 'board eraser', 277: 'rice', 278: 'tape measure/ruler', 279: 'dumbbell', 280: 'hamimelon', 281: 'stapler', 282: 'camel', 283: 'lettuce', 284: 'goldfish', 285: 'meat balls', 286: 'medal', 287: 'toothpaste', 288: 'antelope', 289: 'shrimp', 290: 'rickshaw', 291: 'trombone', 292: 'pomegranate', 293: 'coconut', 294: 'jellyfish', 295: 'mushroom', 296: 'calculator', 297: 'treadmill', 298: 'butterfly', 299: 'egg tart', 300: 'cheese', 301: 'pig', 302: 'pomelo', 303: 'race car', 304: 'rice cooker', 305: 'tuba', 306: 'crosswalk sign', 307: 'papaya', 308: 'hair drier', 309: 'green onion', 310: 'chips', 311: 'dolphin', 312: 'sushi', 313: 'urinal', 314: 'donkey', 315: 'electric drill', 316: 'spring rolls', 317: 'tortoise/turtle', 318: 'parrot', 319: 'flute', 320: 'measuring cup', 321: 'shark', 322: 'steak', 323: 'poker card', 324: 'binoculars', 325: 'llama', 326: 'radish', 327: 'noodles', 328: 'yak', 329: 'mop', 330: 'crab', 331: 'microscope', 332: 'barbell', 333: 'bread/bun', 334: 'baozi', 335: 'lion', 336: 'red cabbage', 337: 'polar bear', 338: 'lighter', 339: 'seal', 340: 'mangosteen', 341: 'comb', 342: 'eraser', 343: 'pitaya', 344: 'scallop', 345: 'pencil case', 346: 'saw', 347: 'table tennis paddle', 348: 'okra', 349: 'starfish', 350: 'eagle', 351: 'monkey', 352: 'durian', 353: 'game board', 354: 'rabbit', 355: 'french horn', 356: 'ambulance', 357: 'asparagus', 358: 'hoverboard', 359: 'pasta', 360: 'target', 361: 'hotair balloon', 362: 'chainsaw', 363: 'lobster', 364: 'iron', 365: 'flashlight'}
o365_category2label = {k: i for i, k in enumerate(o365_category2name.keys())}
o365_label2category = {v: k for k, v in o365_category2label.items()}

if __name__ == '__main__':
    img_folder = './dataset/objects365_v1/train/'
    ann_file = './dataset/objects365_v1/annotations/objects365_train.json'
    transforms = None
    return_masks = False
    remap_mscoco_category = False

    objects365_ds = Objects365Detection(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)

    img, target = objects365_ds[0]