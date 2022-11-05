# CNN-relation-extraction

Mindspore Implementation of [Relation Classification via Convolutional Deep Neural Network](https://www.aclweb.org/anthology/C14-1220.pdf).

## Environment Requirements
* python 3.7.5
* mindspore-dev 1.9.0 linux

## Data
* [SemEval2010 Task8](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50) \[[paper](https://www.aclweb.org/anthology/S10-1006.pdf)\]
* [Embedding - Turian et
al.(2010)](http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz) \[[paper](https://www.aclweb.org/anthology/P10-1040.pdf)\]

## Usage
1. Download the embedding and decompress it into the `embedding` folder.
2. Run the following the commands to start the program.
```shell
python run.py
```
More details can be seen by `python run.py -h`.

3. You can use the official scorer to check the final predicted result.
```shell
perl semeval2010_task8_scorer-v1.2.pl proposed_answer.txt predicted_result.txt >> result.txt
```

## Result
`!Not finished yet. There are still problems with dataloader and  f1 value calculation.`


*Note*:
* Some settings are different from those mentioned in the paper.
* No validation set used during training.
* Just complete the part without lexical level features. More detail is available in Section 5.3 in this paper.
* Although I try to set random seeds, it seems that the results of each run are a little different.

## Reference Link
* https://github.com/ShomyLiu/pytorch-pcnn
* https://github.com/FrankWork/conv_relation
* https://github.com/onehaitao/CNN-relation-extraction