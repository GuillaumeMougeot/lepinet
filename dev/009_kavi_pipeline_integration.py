import numpy as np
import torch
import logging
from functools import partial
from fastai.vision.all import load_learner, CategoryMap

def gen_level_idx(vocab, hierarchy):
    """
    Returns a list of integers of the size of vocab indicating the hierarchical level of the taxa at index i.
    - Species is level 0, Genus 1, Family 2, etc.
    - Missing values are noted with -1.

    Args:
    - vocab (list): List of taxa names to find levels for.
    - hierarchy (dict): Nested dictionary representing taxonomic hierarchy.

    Returns:
    - np.ndarray: Array of level indices for each taxa in vocab.
    """
    level_lookup = {}

    def traverse(node, level=0):
        """Recursively traverse the hierarchy and store levels."""
        for key, subnode in node.items():
            level_lookup[key] = level  # Assign level to the taxon
            if isinstance(subnode, dict):
                traverse(subnode, level + 1)
            elif isinstance(subnode, list):  # Leaf nodes (species level)
                for species in subnode:
                    level_lookup[species] = level + 1

    # Build the level lookup dictionary
    traverse(hierarchy)  # Start from -1 so species end up at level 0

    # Assign levels to vocab, default to -1 if missing
    indices = np.array([level_lookup.get(v, -1) for v in vocab], dtype=int)

    # Invert the indices, so species is 0, genus is 1 etc
    # indices = np.where(indices < 0, indices, indices.max()-indices)

    # Warning for missing values
    missing_count = np.sum(indices == -1)
    if missing_count > 0:
        print(f"[Warning] Missing values in taxa dictionary: {missing_count}.")

    return indices

def get_pred_conf(preds:torch.Tensor, vocab:CategoryMap, indices:np.ndarray):
    """Returns predicted labels and confidence for each pred and for each 
    hierarchy level.

    `preds` is a batch of predictions.
    """
    out_preds = []
    out_confs = []
    indices = torch.from_numpy(indices)
    for i in range(indices.max()+1):
        one_level_pred = preds[:,indices==i].cpu().numpy()
        one_level_prd = vocab[indices==i][one_level_pred.argmax(axis=1)]
        one_level_cnf = one_level_pred.max(axis=1)
        out_preds += [one_level_prd]
        out_confs += [one_level_cnf]
    return np.array(out_preds).swapaxes(0,1), np.array(out_confs).swapaxes(0,1)

class FastaiSpeciesClassifier:
    def __init__(self, speciesModelPath, device):
        self.log = logging.getLogger(__name__)
        self.log.info("Moth species model path %s", speciesModelPath)
        # Load fastai Learner instead of previous speciesClassifier
        self.speciesLearner = load_learner(speciesModelPath, cpu=(device == 'cpu'))
        self.speciesLearner.model.eval()

        indices = gen_level_idx(
            self.speciesLearner.dls.vocab,
            self.speciesLearner.hierarchy)
    
        self.get_pred_conf = partial(
            get_pred_conf, 
            vocab=self.speciesLearner.dls.vocab,
            indices=indices,)

    def extractCrop(self, image, bbox):
        if image is None:
            print("Error image cannot be none")
            raise Exception("None image in extract crop")
        x1 = bbox.x1
        x2 = bbox.x2
        y1 = bbox.y1
        y2 = bbox.y2
        image_crop = image[y1:y2, x1:x2]
        return image_crop

    def batchFromDetections(self, image, detections):
        if image is None:
            print("Image must not be None")
            raise Exception("None image not allowed in batchFromDetections")
        print(f"Batching {len(detections)} detections")
        print(f"Batching {detections} detections")
        imagesInBatch = []
        detection_ids = []
        for idx in range(len(detections)):
            print(f"Adding: {detections[idx]}")
            bbox = detections[idx].bbox
            im = self.extractCrop(image, bbox)
            imagesInBatch.append(np.array(im))
            detection_ids.append(detections[idx].id)
        return { "imagesInBatch": imagesInBatch, "detection_ids": detection_ids }

    def classifySpeciesBatch(self, batch):
        detection_ids = batch["detection_ids"]
        images = batch["imagesInBatch"]

        # Create fastai test dataloader
        test_dl = self.speciesLearner.dls.test_dl(images)

        # Inference without progress bar or logging
        with self.speciesLearner.no_bar(), self.speciesLearner.no_logging():
            preds, _ = self.speciesLearner.get_preds(dl=test_dl)
        
        # Get predictions classes and confidence
        prds, cnfs = self.get_pred_conf(preds)

        results = []
        for idx, (prd, cnf) in enumerate(zip(prds, cnfs)):
            results.append({
                "id": detection_ids[idx],
                "label": prd,
                "labelId": None,
                "confidence_value": cnf
            })

        return results


if __name__=="__main__":
    pass





