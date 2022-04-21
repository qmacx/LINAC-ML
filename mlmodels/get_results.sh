#!/usr/bin/sh

> results.txt

#python3 pmq_dnn.py >> results.txt &&
#python3 qms_dnn.py >> results.txt &&
#python3 beamline_dnn.py >> results.txt &&
#python3 qms_dnn_otrdrop.py >> results.txt &&
python3 finalclf.py >> results.txt &&
python3 finalregression.py >> results.txt &&
python3 clfuncertainty.py >> results.txt &&
python3 regressionuncertainty.py >> results.txt &&

cat results.txt
