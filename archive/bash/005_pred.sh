#!/bin/sh

# python dev/005_lepi_large_pred.py\
#  -i /home/george/codes/lepinet/data/flemming/images\
#  -m /home/george/codes/lepinet/data/lepi/models/04-lepi-prod_model1\
#  -hp /home/george/codes/lepinet/data/lepi/hierarchy_all.json\
#  -o /home/george/codes/lepinet/data/flemming/preds_04-lepi-prod_model1-1.csv

python dev/005_lepi_large_pred.py\
 -i /home/george/codes/lepinet/data/flemming/images\
 -m /home/george/codes/lepinet/data/lepi/models/20250424-lepi-prod_model1-save\
 -hp /home/george/codes/lepinet/data/lepi/hierarchy_all.json\
 -o /home/george/codes/lepinet/data/flemming/preds_20250424-lepi-prod_model1-save.csv