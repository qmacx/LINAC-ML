#!/usr/bin/sh

> results.txt

python3 pmq_dnn.py >> results.txt &&
python3 qms_dnn.py >> results.txt &&
python3 beamline_dnn.py >> results.txt &&
python3 qms_dnn_otrdrop.py >> results.txt &&
python3 finalmodel.py >> results.txt &&

cat results.txt
