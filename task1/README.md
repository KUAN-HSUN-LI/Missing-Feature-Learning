# Missing-Feature-Learning-task1

- 11/23
  - counter, plot_box, get_outlier
- 11/24

  - implementing simpleNet
  - | simpleNet | Train   | Valid   | Public  |
    | --------- | ------- | ------- | ------- |
    |           | 0.52892 | 0.51984 | 0.54092 |

  - parameters
    - epoch=47
    - batch_size=256
    - loss:f1_loss
    - opt:Adam -lr=2e-4
    - baseline model 1(training without F1 data)

- 11/25
  - using CE for simpleNet
  - | simpleNet | Train | Valid | Public |
    | --------- | ----- | ----- | ------ |
    |           | 0.694 | 0.673 | 0.649  |
  - parameters
    - epoch=397
    - batch_size=1024
    - loss:CrossEntropy
    - opt:Adam -lr=1e-4
    - baseline model 1(training without F1 data)
