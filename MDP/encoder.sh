#! /bin/bash
if [$3 != ""] ; then
	python3 encoder_stoch.py $1 $2  
else
	python3 encoder.py $1
fi 