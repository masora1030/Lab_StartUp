## 自作のネットワークとVGG16に対して、MNISTのデータセットを用いて、学習曲線や性能の比較を行った。OptimizerはMomentumSGD
### My Many CNN model : (Conv BN Relu | Conv BN Relu (dropout) (pooling) | Conv BN Relu | Conv BN Relu (dropout) | Conv BN Relu | Conv BN Relu | Affine relu | Affine (Softmax))
### My Few CNN model : (Conv BN Relu Pooling | Conv BN Relu (dropout) | Affine relu | Affine (Softmax))

| Network   |      test_accuracy      |  learning time [sec] |
|----------|:-------------:|------:|
| My Few CNN |  75.28 % | 295.004 |
| My Many CNN |  78.09 % | 390.053 |
| VGG16 |    83.15 %  |   793.937 |

- ネットワークをVGG16にすると、自作のCNN多層ネットワークよりもaccが格段に向上した。学習時間は2倍ほどになった。
- 層が手薄だとたくさん過学習した。
- 性能は学習時間とtest acc(汎化性能)とのトレードオフ？
