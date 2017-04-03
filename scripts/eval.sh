#!/bin/sh

code=../hyperwords
src_word_model=model_word_$1.txt
src_context_model=model_context_$1.txt
model=wordrank

cp $src_word_model $model.words
cp $src_context_model $model.contexts

python $code/hyperwords/text2numpy.py $model.words
python $code/hyperwords/text2numpy.py $model.contexts

echo "WS353 Results"
echo "-------------"
python $code/hyperwords/ws_eval.py embedding $model $code/testsets/ws/ws353.txt
echo

echo "Google Analogy Results"
echo "----------------------"
python $code/hyperwords/analogy_eval.py --w+c embedding $model $code/testsets/analogy/google.txt
echo 
