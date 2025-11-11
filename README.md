# lepinet
Lepidoptera Classification model.

## File structure:

In bold are production-ready script.

* 000: ipynb, first and second version of the model training.
* 001: ipynb, first version of the prediction process, including some result data exploration.
* 002: ipynb, result exploration and interpretation.
* **003**: py, first version of an actual production-level training script.
* 004: ipynb, formatting of test dataset - flemming dataset.
* **005**: py, first version of an prediction workflow with the model trained with 003.
* 006: ipynb, results exploration and interpretation on the test dataset.
* 007: ipynb, dev notebook for integrating fastai model inside AMI pipeline.
* 008: py, first version of pipeline for fastai model integration in AMI pipeline.
* **009**: py, production-ready script for integration of fastai model in AMI pipeline.
* 010: py, experimentation with UCloud communication.
* **011**: py, second version of production-level training script.
* **012**: py, test script of the lepi model
* 013: ipynb, plot some curves about the classifier

## Unorganized TODOs

* Change the head of the model with the hierarchical approach?
* Multi-GPU training? with bigger batch size? If this could half the training time that would be great.
* How does this global model perform on the eu_lepi test set? on the global_lepi test set? on flemming test set?
* Profile the training steps.