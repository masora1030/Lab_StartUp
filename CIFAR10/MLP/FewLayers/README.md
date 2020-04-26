## CIFAR10のデータセットに対して、3層のMLPを構成した。
### (Affine relu (dropout) | Affine relu (dropout) | Affine (Softmax))
- 隠れ層のunit数は入力と同サイズの1024(32*32)入れた時が一番学習率が良かった。
