#!/bin/sh

python dev/005_lepi_large_pred.py\
 -i /home/george/codes/lepinet/data/to_pred/in\
 -m /home/george/codes/lepinet/data/lepi/models/04-lepi-prod_model1-save-hierarchy-id2name\
 -hp /home/george/codes/lepinet/data/lepi/hierarchy_all.json\
 -o /home/george/codes/lepinet/data/to_pred/out/screenshot.csv