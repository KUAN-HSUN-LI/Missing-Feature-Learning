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
- model 1 (11/28)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.84980 | 0.78462 | 0.78536 |
    - parameters
      - epoch: 170
      - batch_size: 128
      - loss: CELoss
      - opt: Adam -lr=1e-3, step 0.1 per 150
      
- model 1 (12/2)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.86145 | 0.79423 | 0.79793 |
    - parameters
      - epoch: 300
      - batch_size: 128
      - loss: BCEWithLogitsLoss
      - opt: Adam -lr=1e-3, step 0.5 per 100


### baseline2
- model 1 (11/28)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.80778 | 0.78269 | 0.77907 |
    - parameters
      - epoch: stage1 200, stage2 200
      - batch_size: 128
      - loss: CELoss
      - opt: Adam -lr=1e-3, step 0.5 per 50

      
### baseline3
- model 1 (11/28)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.83611 | 0.79231 | 0.78311 |
    
    - parameters
      - epoch: 200
      - batch_size: 64
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


### GAN
- model 1 with noise (12/3)
    - | simpleNet | Train   | Valid   | Public  |
      | --------- | ------- | ------- | ------- |
      |           | 0.83634 | 0.81145 | 0.80466 |
    
    - parameters
      - epoch: 800
      - batch_size: 32
      - loss: BCEWithLogitsLoss
      - opt: Adam 
      - lr: 
        - model: 1e-3, step 0.5 per 150
        - generator/discriminator: 1e-4, step 0.5 per 300