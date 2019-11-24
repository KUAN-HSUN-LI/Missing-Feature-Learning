# Missing-Feature-Learning-task2

```
cd task2/src/baseline1/
```

1. Train:
```
python main.py --arch simpleNet1 --do_train
```

2. Predict:
```
python main.py --arch simpleNet1 --do_predict --ckpt 47
```

- 11/24
  - implementing simpleNet
  - | simpleNet | Train   | Valid   | Public  |
    | --------- | ------- | ------- | ------- |
    |           | 0.58991 | 0.59808 | 0.57880 |

  - parameters
    - epoch=47
    - batch_size=256
    - loss:f1_loss
    - opt:Adam -lr=5e-5
    - baseline model 1(training without F2, F7, F12 data)
