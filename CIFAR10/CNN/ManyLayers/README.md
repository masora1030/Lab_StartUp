## CIFAR10のデータセットに対して、8層のCNNを構成した。
### (Conv (BN) Relu | Conv (BN) Relu (dropout) (pooling) | Conv (BN) Relu | Conv (BN) Relu (dropout) | Conv (BN) Relu | Conv (BN) Relu | Affine relu | Affine (Softmax))
- OptimizerはMomentunSGDで、convフィルタのサイズは3×3で、チャンネル数は3->64->64->128->128->256->256と増やしていった。poolingは2×2。dropoutは0.2
- 4層の時と比べてtest accは74%(+2.0%)と微増したが、思ったより変わらなかった。。。25~50epochを境に過学習している。
- Batch Normalizationを各アクティベート関数の前に追加したところ、過学習をある程度抑えられ、性能が飛躍的に上がり、tet accが80%となった。若干、学習時間は長くなった。
