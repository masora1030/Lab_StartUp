## いくつかのOptimizerに対して、MNISTのデータセットとVGG16ネットワークを用いて、学習曲線や性能の比較を行った。

| Optimizer   |      test_accuracy      |  learning time [sec] |
|----------|:-------------:|------:|
| SGD |  99.44 % | 843.484 |
| MomentumSGD |  99.08 % | 879.802 |
| Adam |    99.35 %  |   946.819 |
| RMSprop | 99.47 %  |    898.542 |
| Adamax | 99.34 % |    996.010 |


- 速くて高精度なSGD
- Adamは遅い。後lossの学習曲線がガタガタしてる。
- MNISTに対してはOptimizerによってあんまり差がない
