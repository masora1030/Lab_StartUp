## 自作のネットワークとVGG16に対して、MNISTのデータセットを用いて、学習曲線や性能の比較を行った。OptimizerはMomentumSGD
### My Many CNN model : (Conv BN Relu | Conv BN Relu (dropout) (pooling) | Conv BN Relu | Conv BN Relu (dropout) | Conv BN Relu | Conv BN Relu | Affine relu | Affine (Softmax))
### My Few CNN model : (Conv BN Relu Pooling | Conv BN Relu (dropout) | Affine relu | Affine (Softmax))

| Network   |      test_accuracy      |  learning time [sec] |
|----------|:-------------:|------:|
| My Few CNN |  99.08 % | 260.276 |
| My Many CNN |  99.46 % | 351.942 |
| VGG16 |    99.34 %  |   872.170 |

- ネットワークをVGG16にすると、学習時間は3倍ほどになった。精度はそこまで。
- 層が適当で手薄設計でもちゃんと精度出る。データセットが単純だからかな。(カラーバリエーションなど)
- 自作のMy Many CNNが一番か学習も抑えられて精度も出てて良かった！！(VGG16はImageNet用のモデルなのでそれはそうかもしれないが)
