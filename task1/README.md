# Missing-Feature-Learning-task1

- 11/23
  - counter, plot_box, get_outlier
- 12/10

  - baseline1
    | simpleNet | Train | Valid | Public |
    | --------- | ------- | ------- | ------- |
    | | 0.91213 | 0.80843 | 0.77573 |

    parameters

    - epoch=2694
    - batch_size=256
    - loss:CE
    - opt:Adam -lr=1e-4

  - simpleGAN
    | simpleNet | Train | Valid | Public |
    | --------- | ------- | ------- | ------- |
    | | 0.92395 | 0.81130 | 0.73720 |

    parameters

    - epoch=3135
    - batch_size=256
    - loss: cls - CE
    - opt:Adam -lr=1e-4
