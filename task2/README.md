# Missing-Feature-Learning-task2

## How to run
```
cd task2/src/baseline1/
```

1. Train:
```
python main.py --arch baseline1 --do_train
```

2. Predict:
```
python main.py --arch baseline1 --do_predict --ckpt 300
```

## Experiments

### simpleNet without any missing feature
- model 1 (11/25)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.94911 | 0.85865 |   N/A   |


### baseline1
- model 1 (11/24)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.87535 | 0.78365 |   N/A   |
    - parameters
      - epoch: 200
      - batch_size: 32
      - loss: CELoss
      - opt: Adam -lr=1e-3, step 0.1 per 100


### baseline2
- model 1 (11/24)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.87535 | 0.78365 |   N/A   |
    - parameters
      - epoch: 200
      - batch_size: 32
      - loss: CELoss
      - opt: Adam -lr=1e-3, step 0.1 per 100

      
### baseline3
- model 1 (11/27)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.82788 | 0.79904 |   N/A   |
    
    - parameters
      - epoch: 300
      - batch_size: 32
      - loss: CELoss, MSELoss
      - opt: Adam -lr=1e-3, step 0.5 per 100
          

### trial 1
- model 1 (11/25)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.84445 | 0.80673 | 0.78536 |
    
    - parameters
      - epoch: 300
      - batch_size: 128
      - loss: CELoss, MSELoss
      - opt: Adam -lr=1e-3, step 0.1 per 100
