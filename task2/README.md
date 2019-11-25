# Missing-Feature-Learning-task2

```
cd task2/src/baseline1/
```

1. Train:
```
python main.py --arch baseline1 --do_train
```

2. Predict:
```
python main.py --arch baseline1 --do_predict --ckpt 210
```

- simpleNet without any missing feature
- | simpleNet | Train   | Valid   | Public  |
  | --------- | ------- | ------- | ------- |
  |           | 0.94911 | 0.85865 |   N/A   |


- 11/24 baseline1 model 1
- | simpleNet | Train   | Valid   | Public  |
  | --------- | ------- | ------- | ------- |
  |           | 0.87535 | 0.78365 |   N/A   |
- parameters
  - epoch: 200
  - batch_size: 32
  - loss: CELoss
  - opt: Adam -lr=1e-3, step 0.1 per 100
  - baseline model 1(training without F2, F7, F12 data)


- 11/25 baseline2 model 1
- | simpleNet | Train   | Valid   | Public  |
  | --------- | ------- | ------- | ------- |
  |           | 0.84445 | 0.80673 | 0.78536 |

- parameters
  - epoch: 300
  - batch_size: 128
  - loss: CELoss, MSELoss
  - opt: Adam -lr=1e-3, step 0.1 per 100
  - baseline model 1(training without F2, F7, F12 data)