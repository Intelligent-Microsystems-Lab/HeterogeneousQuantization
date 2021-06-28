

```
sudo apt install subversion
svn export https://github.com/google-research/google-research/trunk/kws_streaming
pip install --user --upgrade pip
pip install --user -r kws_streaming/requirements.txt
```

```
I0615 00:14:02.404654 140432697560128 train.py:226] step: 23400, train_loss: 1.4312, train_accuracy: 0.8919, validation_loss: 1.3657, validation_accuracy: 0.9116
I0615 00:15:05.904953 140432697560128 train.py:253] FINAL LOSS 1.3657, FINAL ACCURACY: 0.9116 on TEST SET
```