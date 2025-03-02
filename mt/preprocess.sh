#!/usr/bin/bash

# ################################################################################
# # from 'https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh'
# ################################################################################

if [ ! -d data ]; then
    mkdir data
fi
cd data

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

#URL="http://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz"
#URL="https://drive.google.com/u/0/uc?id=1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed&export=download"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=prep
tmp=prep/tmp
orig=orig

mkdir -p $orig $tmp $prep
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed" -O 2014-01.tgz && rm -rf /tmp/cookies.txt
tar zxvf 2014-01.tgz
mv 2014-01/texts/de/en/$GZ $orig


echo "Downloading data from ${URL}..."
cd $orig
#wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 50
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $prep/valid.de-en.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $prep/train.de-en.$l

    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $prep/test.de-en.$l
done
cd ..

################################################################################
# encode data into pytorch format
################################################################################
srclang="de"
tgtlang="en"
python preprocess.py \
    -train_src data/${prep}/train.${srclang}-${tgtlang}.${srclang} \
    -train_tgt data/${prep}/train.${srclang}-${tgtlang}.${tgtlang} \
    -valid_src data/${prep}/valid.${srclang}-${tgtlang}.${srclang} \
    -valid_tgt data/${prep}/valid.${srclang}-${tgtlang}.${tgtlang} \
    -test_src data/${prep}/test.${srclang}-${tgtlang}.${srclang} \
    -test_tgt data/${prep}/test.${srclang}-${tgtlang}.${tgtlang} \
    -save_data data/iwslt14 \
    -src_min_freq 3 \
    -tgt_min_freq 3 \
    -lower 
