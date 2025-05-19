import logging
import random
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import cv2
import numpy as np
import torch
from fastai.vision.all import CategoryMap, load_learner


ERDA_MODELS = "https://anon.erda.au.dk/share_redirect/C1nJdS1jtA/{}"
DEFAULT_MODEL = "00_eulepi.pkl"
MODEL_LOCAL_PATH="{}"

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
    indices = np.where(indices < 0, indices, indices.max()-indices)

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
    valid_type = ['all', 'species', 'best']
    f"""
    Prediciton class for the species classifier trained with fastai.

    Args:
    - speciesModelPath (str): Path to the species model, if None download it from a ERDA link.
    - device (str): Device to run the computations. Either cpu or cuda.
    - output_type (str): Type of the output: one of {valid_type}. 'all' outputs a list of the best model prediction per hierarchy level, in the following order: species, genus, family. 'species' only outputs the species level. 'best' only outputs the lowest ranked predictions with a confidence above `th`.
    - th (float): Confidence threshold.
    """
    def __init__(self, speciesModelPath:str=None, device='cuda', output_type: str='species', th: float=0.5):
        assert output_type in self.valid_type, f"Error: `output_type` must be one of {self.valid_type} but found {output_type}"

        self.log = logging.getLogger(__name__)

        # Download model from ERDA if not found locally
        if speciesModelPath is None:
            speciesModelPath = MODEL_LOCAL_PATH.format(DEFAULT_MODEL)
        if not Path(speciesModelPath).is_file() or\
            not Path(DEFAULT_MODEL).exists():
            self.log.info(f"Model not found. Downloading from {ERDA_MODELS.format(DEFAULT_MODEL)}")
            url = ERDA_MODELS.format(DEFAULT_MODEL)
            with urllib.request.urlopen(url) as response, open(speciesModelPath, 'wb') as out_file:
                out_file.write(response.read())
        else:
            print(f"Found model in {speciesModelPath}")

        self.log.info("Moth species model path %s", speciesModelPath)
        # Load fastai Learner instead of previous speciesClassifier
        self.speciesLearner = load_learner(speciesModelPath, cpu=(device == 'cpu'))
        self.speciesLearner.model.eval()
        self.id2name = self.speciesLearner.id2name

        indices = gen_level_idx(
            self.speciesLearner.dls.vocab,
            self.speciesLearner.hierarchy)
    
        self.get_pred_conf = partial(
            get_pred_conf, 
            vocab=self.speciesLearner.dls.vocab,
            indices=indices,)
    
        self.output_type = output_type
        self.th = th

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
            if self.output_type=='species':
                results.append({
                    "id": detection_ids[idx],
                    "label": self.id2name[prd[0]],
                    "labelId": prd[0],
                    "confidence_value": cnf[0]
                })
            elif self.output_type=='best':
                i=0 # Index of when the cnf is above 0.5
                while i < len(cnf) and cnf[i] < self.th: i += 1
                if i == len(cnf): # No prediction, outputs highest level
                    results.append({
                        "id": detection_ids[idx],
                        "label": self.id2name[prd[-1]],
                        "labelId": prd[-1],
                        "confidence_value": cnf[-1]
                    })
                else:
                    results.append({
                        "id": detection_ids[idx],
                        "label": self.id2name[prd[i]],
                        "labelId": prd[i],
                        "confidence_value": cnf[i]
                    })
            elif self.output_type=='all':
                results.append({
                    "id": detection_ids[idx],
                    "label": [self.id2name[p] for p in prd],
                    "labelId": prd,
                    "confidence_value": cnf
                })
            else:
                raise NotImplementedError("Choose a valid output type.")

        return results


def test_species_classifier_with_fake_detections(
    images_folder: str,
    species_model_path: str = None,
    num_detections: int = 3,
    device: str = 'cpu'
):
    """
    Test the FastaiSpeciesClassifier using one image and multiple fake detections.

    Args:
        species_model_path (str): Path to the Fastai .pkl learner.
        images_folder (str): Folder containing test images.
        num_detections (int): Number of fake detections to generate.
        device (str): 'cpu' or 'cuda' for model inference.
    """
    # Load image
    image_files = list(Path(images_folder).glob("*.*"))
    if not image_files:
        raise FileNotFoundError("No images found in the specified folder.")
    img_path = str(image_files[0])
    print(f"Using image: {img_path}")

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    height, width, _ = image.shape

    # Generate fake detections with bounding boxes that cover at least 90% of the image
    detections = []
    for i in range(num_detections):
        pad_x = random.randint(0, int(width * 0.1))
        pad_y = random.randint(0, int(height * 0.1))

        x1 = max(0, pad_x)
        y1 = max(0, pad_y)
        x2 = min(width, width - pad_x)
        y2 = min(height, height - pad_y)

        bbox = SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)
        detection = SimpleNamespace(id=f'fake_detection_{i+1}', bbox=bbox)
        detections.append(detection)

    # Load classifier and process detections
    classifier = FastaiSpeciesClassifier(species_model_path, device=device, output_type='best')

    batch = classifier.batchFromDetections(image, detections)
    results = classifier.classifySpeciesBatch(batch)

    # Print results
    for res in results:
        print("\n--- Detection Result ---")
        print("Detection ID:", res["id"])
        print("Predicted Labels (Hierarchy):", res["label"])
        print("Predicted Labels ID (Hierarchy):", res["labelId"])
        print("Confidence Scores:", res["confidence_value"])
   

if __name__=="__main__":
    test_species_classifier_with_fake_detections(
        # species_model_path="/home/george/codes/lepinet/data/lepi/models/04-lepi-prod_model1-save-hierarchy-id2name",
        images_folder="/home/george/codes/lepinet/data/lepi/images/1732680",
        num_detections = 5,
        device= 'cuda'
    )





