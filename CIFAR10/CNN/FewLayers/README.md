## CIFAR10のデータセットに対して、4層のCNNを構成した。
### (Conv Relu Pooling | Conv Relu (dropout) | Affine relu | Affine (Softmax))
- OptimizerはMomentunSGD、convフィルタのサイズは5×5で、チャンネル数は3->27->54と増やしていった。poolingは2×2
- 最後の出力手前のAffineでの変換サイズを108,256,1024の一次元データそれぞれにして試したところ、1024にしたものの方が(おそらく表現力が高く)精度が向上したので、1024にした。
- 上記のハイパラ設定でのモデルの最終的なtest accは72%程度であった。
- convフィルタのサイズを3×3にしてもあんまり精度には変化なし
