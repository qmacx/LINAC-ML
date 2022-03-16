#!/usr/bin/sh

> results.txt

python3 mlmodels/pmq_dnn.py >> results.txt &&
python3 mlmodels/qms_dnn.py >> results.txt &&
python3 mlmodels/beamline_dnn.py >> results.txt &&
python3 mlmodels/qms_dnn_otrdrop.py >> results.txt &&
python3 mlmodels/finalmodel.py >> results.txt &&

cat results.txt
