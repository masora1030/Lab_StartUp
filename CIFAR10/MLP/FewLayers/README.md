## CIFAR10のデータセットに対して、3層のMLPを構成した。
### (Affine relu (dropout) | Affine relu (dropout) | Affine (Softmax))
- 隠れ層のunit数は、入力と同サイズの1024(32*32)を入れた時が一番学習率が良かった。
- 隠れ層のunit数は、2048などを入れるとおかしくなった。(うまく学習しない。)
