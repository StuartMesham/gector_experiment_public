#/bin/bash

wget https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt

mkdir -p data
wget https://github.com/grammarly/gector/raw/master/data/verb-form-vocab.txt -P data

mkdir -p data_downloads
cd data_downloads

# NUCLE
# <NUCLE DOWNLOAD COMMAND HERE>
# Your command should download a file called release3.3.tar.bz2
tar -xvf release3.3.tar.bz2
rm release3.3.tar.bz2
mkdir nucle
mv release3.3/bea2019/nucle.train.gold.bea19.m2 nucle
rm -r release3.3

# Locness
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
tar -xzf wi+locness_v2.1.bea19.tar.gz
rm wi+locness_v2.1.bea19.tar.gz

# FCE
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
tar -xzf fce_v2.1.bea19.tar.gz
rm fce_v2.1.bea19.tar.gz

# PIE Synthetic
gdown --id 1bl5reJ-XhPEfEaPjvO45M7w0yN-0XGOA
unzip synthetic.zip -d synthetic
rm synthetic.zip

# Lang8
gdown --id 148M_4LvHyb0J_sNWxrFSYjv3lO5ACigK
mkdir -p lang8
tar -xzf lang8.bea19.tar.gz --directory lang8
rm lang8.bea19.tar.gz

# CoNLL-2014 Test Set
wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar -xzf conll14st-test-data.tar.gz
rm conll14st-test-data.tar.gz

cd ..

python utils/test_train_split.py --f1 data_downloads/synthetic/a1/a1_train_incorr_sentences.txt --f2 data_downloads/synthetic/a1/a1_train_corr_sentences.txt

mkdir -p datasets/unprocessed/stage_1
cp data_downloads/synthetic/a1/a1_train_incorr_sentences.train.txt datasets/unprocessed/stage_1/out_uncorr.train.txt
cp data_downloads/synthetic/a1/a1_train_corr_sentences.train.txt datasets/unprocessed/stage_1/out_corr.train.txt
cp data_downloads/synthetic/a1/a1_train_incorr_sentences.dev.txt datasets/unprocessed/stage_1/out_uncorr.dev.txt
cp data_downloads/synthetic/a1/a1_train_corr_sentences.dev.txt datasets/unprocessed/stage_1/out_corr.dev.txt

# stage 2

# get parallel sentences for lang-8
python utils/corr_from_m2.py -out data_downloads/lang8/out_corr.txt -id 0 data_downloads/lang8/lang8.train.auto.bea19.m2
python utils/uncorr_from_m2.py -out data_downloads/lang8/out_uncorr.txt data_downloads/lang8/lang8.train.auto.bea19.m2
python utils/test_train_split.py --f1 data_downloads/lang8/out_uncorr.txt --f2 data_downloads/lang8/out_corr.txt

# get parallel sentences for NUCLE
python utils/corr_from_m2.py -out data_downloads/nucle/out_corr.txt -id 0 data_downloads/nucle/nucle.train.gold.bea19.m2
python utils/uncorr_from_m2.py -out data_downloads/nucle/out_uncorr.txt data_downloads/nucle/nucle.train.gold.bea19.m2
python utils/test_train_split.py --f1 data_downloads/nucle/out_uncorr.txt --f2 data_downloads/nucle/out_corr.txt


# get parallel sentences for FCE
python utils/corr_from_m2.py -out data_downloads/fce/out_corr.train.txt -id 0 data_downloads/fce/m2/fce.train.gold.bea19.m2
python utils/uncorr_from_m2.py -out data_downloads/fce/out_uncorr.train.txt data_downloads/fce/m2/fce.train.gold.bea19.m2

python utils/corr_from_m2.py -out data_downloads/fce/out_corr.dev.txt -id 0 data_downloads/fce/m2/fce.dev.gold.bea19.m2
python utils/uncorr_from_m2.py -out data_downloads/fce/out_uncorr.dev.txt data_downloads/fce/m2/fce.dev.gold.bea19.m2

# get parallel sentences for lang-8
python utils/corr_from_m2.py -out data_downloads/lang8/out_corr.txt -id 0 data_downloads/lang8/lang8.train.auto.bea19.m2
python utils/uncorr_from_m2.py -out data_downloads/lang8/out_uncorr.txt data_downloads/lang8/lang8.train.auto.bea19.m2
python utils/test_train_split.py --f1 data_downloads/lang8/out_uncorr.txt --f2 data_downloads/lang8/out_corr.txt

# get parallel sentences for wi+locness
python utils/corr_from_m2.py -out data_downloads/wi+locness/out_corr.train.txt -id 0 data_downloads/wi+locness/m2/ABC.train.gold.bea19.m2
python utils/uncorr_from_m2.py -out data_downloads/wi+locness/out_uncorr.train.txt data_downloads/wi+locness/m2/ABC.train.gold.bea19.m2

python utils/corr_from_m2.py -out data_downloads/wi+locness/out_corr.dev.txt -id 0 data_downloads/wi+locness/m2/ABCN.dev.gold.bea19.m2
python utils/uncorr_from_m2.py -out data_downloads/wi+locness/out_uncorr.dev.txt data_downloads/wi+locness/m2/ABCN.dev.gold.bea19.m2

# create combined dev and train sets for stage 2
mkdir -p datasets/unprocessed/stage_2
cat data_downloads/nucle/out_corr.train.txt data_downloads/fce/out_corr.train.txt data_downloads/lang8/out_corr.train.txt data_downloads/wi+locness/out_corr.train.txt > datasets/unprocessed/stage_2/out_corr.train.txt
cat data_downloads/nucle/out_uncorr.train.txt data_downloads/fce/out_uncorr.train.txt data_downloads/lang8/out_uncorr.train.txt data_downloads/wi+locness/out_uncorr.train.txt > datasets/unprocessed/stage_2/out_uncorr.train.txt

cat data_downloads/nucle/out_corr.dev.txt data_downloads/fce/out_corr.dev.txt data_downloads/lang8/out_corr.dev.txt data_downloads/wi+locness/out_corr.dev.txt > datasets/unprocessed/stage_2/out_corr.dev.txt
cat data_downloads/nucle/out_uncorr.dev.txt data_downloads/fce/out_uncorr.dev.txt data_downloads/lang8/out_uncorr.dev.txt data_downloads/wi+locness/out_uncorr.dev.txt > datasets/unprocessed/stage_2/out_uncorr.dev.txt

mkdir -p datasets/unprocessed/stage_3
cp data_downloads/wi+locness/out_uncorr.train.txt datasets/unprocessed/stage_3/out_uncorr.train.txt
cp data_downloads/wi+locness/out_corr.train.txt datasets/unprocessed/stage_3/out_corr.train.txt
cp data_downloads/wi+locness/out_uncorr.dev.txt datasets/unprocessed/stage_3/out_uncorr.dev.txt
cp data_downloads/wi+locness/out_corr.dev.txt datasets/unprocessed/stage_3/out_corr.dev.txt

# CoNLL-2014 Test Set
python utils/uncorr_from_m2.py -out data_downloads/conll14st-test-data/out_uncorr.test.txt data_downloads/conll14st-test-data/noalt/official-2014.combined.m2
