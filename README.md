# LGAN
This repository collects chainer implementation of [LGAN](http://proceedings.mlr.press/v97/tao19a.html).
These codes are built based on the [chainer-gan-lib](https://github.com/pfnet-research/chainer-gan-lib).

How to use
-------

Install the requirements first:
```
pip install -r requirements.txt
```
This implementation has been tested with the following versions.
```
python 3.5.2
chainer 4.0.0
+ https://github.com/chainer/chainer/pull/3615
+ https://github.com/chainer/chainer/pull/3581
cupy 3.0.0
tensorflow 1.2.0 # only for downloading inception model
numpy 1.11.1
```
Download the inception score module [here](https://github.com/sdai654416/LGAN/tree/master/common/inception).
```
git submodule update -i
```
Download the inception model.
```
cd common/inception
python download.py --outfile inception_score.model
```
You can start training with `train.py`.

Please see `example.sh` to train other algorithms.

Inception scores are calculated by average of 10 evaluation with 5000 samples.

FIDs are calculated with 50000 train dataset and 10000 generated samples.




