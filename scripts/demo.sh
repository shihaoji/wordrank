#!/bin/bash

code=../glove

if [ ! -e text8 ]; then
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
rm text8.zip
fi

CORPUS=text8
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence
COOCCURRENCE_SHUF_FILE=wiki.toy
VERBOSE=2
MEMORY=8.0
VOCAB_MIN_COUNT=5
VOCAB_MAX_SIZE=1000000
WINDOW_SIZE=15
MATRIX_DIR=matrix.toy
META_FILE=meta
NUM_CORES=16

#######################
printf "1. clean up\n"
#######################
rm -rf $MATRIX_DIR; mkdir $MATRIX_DIR

##########################################
printf "\n2. build cooccurrence matrix\n"
##########################################
$code/vocab_count -min-count $VOCAB_MIN_COUNT -max-vocab $VOCAB_MAX_SIZE -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$code/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$code/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

########################################################
printf "\n3. prepare the data directory for training\n"
########################################################
mv $COOCCURRENCE_SHUF_FILE $MATRIX_DIR
cut -d' ' -f1,1 $VOCAB_FILE > $MATRIX_DIR/$VOCAB_FILE
rm -rf $COOCCURRENCE_FILE $VOCAB_FILE

numwords=$(wc -l < $MATRIX_DIR/$VOCAB_FILE)
numlines=$(wc -l < $MATRIX_DIR/$COOCCURRENCE_SHUF_FILE)
cat <<EOF > $MATRIX_DIR/$META_FILE
$numwords $numwords
$numlines $COOCCURRENCE_SHUF_FILE
$numwords $VOCAB_FILE
EOF

#########################################################
printf "\n4. wordrank training (take a long time ...)\n"
#########################################################
mpirun -np 1 ../wordrank --path $MATRIX_DIR --nthreads $NUM_CORES --sgd_num 100 --lrate 0.001 --period 20 --iter 500 --epsilon 0.75 --dump_prefix model --dump_period 20 --dim 100 --reg 0 --alpha 100 --beta 99 --loss hinge

#########################################################
printf "\n5. evaluation\n"
#########################################################
./eval 500

