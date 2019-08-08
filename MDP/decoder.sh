#! /bin/bash
if [$4 != ""] ; then
	python3 decoder_stoch.py $1 $2 $3  
else
	python3 decoder.py $1 $2
fi 