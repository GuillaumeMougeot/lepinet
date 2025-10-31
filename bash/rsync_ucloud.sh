#!/bin/sh



rsync -va --exclude=data/ --exclude=.ipynb_checkpoints/ --exclude=__pycache__/ --exclude=.vscode/ --exclude=.git/ --exclude=slurm/ --exclude=logs --exclude=dist/ --exclude=build/ --exclude=biom3d/src/biom3d.egg-info/ /home/george/codes/lepinet ucloud:/work/lepinet