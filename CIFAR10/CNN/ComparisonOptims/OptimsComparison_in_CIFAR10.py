import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import matplotlib.image as img
import time
# CNNの間ではDropoutはあまり挟まない

# データ正規化
form = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize(
                [0.5, 0.5, 0.5],  # RGB 平均
                [0.5, 0.5, 0.5]  # RGB 標準偏差
           )
       ])

# load CIFAR10 data
train_Dataset = dsets.CIFAR10(  # CIFAR10 default dataset
    root='./data_cifar10/',  # rootで指定したフォルダーを作成して生データを展開。これは必須。
    train=True,  # 学習かテストかの選択。これは学習用のデータセット
    transform=form,  # Pytroch のテンソルに変換する  # 自動で形式を変換するのでラベルをOne_hotにする必要等がない。
    download=True)  # ダウンロードするかどうかの選択(当然する) ここには正解ラベルのデータと入力文字のデータの対がまとめて入っている。.

test_dataset = dsets.CIFAR10(
    root='./data_cifar10/',
    train=False, # 学習かテストかの選択。これはテスト用のデータセット
    transform=form,  # Pytroch のテンソルに変換する。 # 自動で形式を変換するのでラベルをOne_hotにする必要等がない。
    download=True)  # ダウンロードするかどうかの選択(当然する)

train_dataset, valid_dataset = torch.utils.data.random_split(  # データセットの分割(validation)
    train_Dataset,  # 分割するデータセット(さっきダウンロードしたやつ)
    [40000, 10000])  # 分割数    (50000個の学習用データを、さらに40000個の学習用データと10000個のバリデーション用データに分割)

print('train_dataset = ', len(train_dataset)) # 40000
print('valid_dataset = ', len(valid_dataset)) # 10000
print('test_dataset = ', len(test_dataset)) # 10000


# set data loader
# データセットからミニバッチ単位でデータを取り出し、ネットワークへ供給するためのローダー(バッチ処理の準備)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,  # データセットの指定
    batch_size=100,  # ミニバッチの指定
    shuffle=True,  # シャッフルするかどうかの指定
    num_workers=2)  # コアの数

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=2)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=2)

# CNN
# CNNモデルの構築。層を増やして8層。
# (Conv BN Relu | Conv BN Relu (dropout) (pooling) | Conv BN Relu | Conv BN Relu (dropout)
# | Conv BN Relu | Conv BN Relu | Affine relu | Affine (Softmax))
# nn.Moduleを継承している。
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # xAT + b (重みとバイアスを足し合わせる作業と、重みパラメータを保持する機能)
        # Conv2d(in_channels, out_channels(), kernel_size(フィルターサイズ), stride=1, padding=0,
        # dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(3, 64, 3) # 32x32x3 -> 30x30x64
        self.bn1 = nn.BatchNorm2d(64)    # Batch Normalization
        self.conv2 = nn.Conv2d(64, 64, 3) # 30x30x64 -> 28x28x64
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        self.dropout1 = nn.Dropout2d(0.2)
        self.pool = nn.MaxPool2d(2, 2)  # 28x28x64 -> 14x14x32
        self.conv3 = nn.Conv2d(64, 128, 3) # 14x14x64 -> 12x12x128
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization
        self.conv4 = nn.Conv2d(128, 128, 3)  # 12x12x128 -> 10x10x128
        self.bn4 = nn.BatchNorm2d(128)  # Batch Normalization
        self.dropout2 = nn.Dropout2d(0.2)
        self.conv5 = nn.Conv2d(128, 256, 3)  # 10x10x128 -> 8x8x256
        self.bn5 = nn.BatchNorm2d(256)  # Batch Normalization
        self.conv6 = nn.Conv2d(256, 256, 3)  # 8x8x256 -> 6x6x256
        self.bn6 = nn.BatchNorm2d(256)  # Batch Normalization
        self.fc1 = nn.Linear(6*6*256, 1024)
        self.bn7 = nn.BatchNorm1d(1024)  # Batch Normalization
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x): # predictに相当(順伝搬)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.dropout1(F.relu(self.bn2(self.conv2(x)))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, 6*6*256)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.fc2(x)
        return x

    # accはoptimizerを用いた学習を記述する中で計算をする。(1エポックごとに)

    # 損失計算の定義は必要ない。(optimizingのcriterionでloss関数に交差エントロピーを指定)

    # 勾配計算の定義も必要ない。(optimizingで指定するcriterionがbackwardを計算してくれる)

# select device
# GPU or CPU? (利用可能ならGPUを使う)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# optimizing
criterion = nn.CrossEntropyLoss() # 交差エントロピーをloss計算に用いる。




# ここから比較のコード



network1 = CNNNet().to(device) # networkにさっき定義したnetworkを代入
optimizer1 = optim.SGD(network1.parameters(), lr=0.01, weight_decay=5e-4) # optimizrにSGDを指定
optKind1 = 'SGD'

network2 = CNNNet().to(device) # networkにさっき定義したnetworkを代入
optimizer2 = optim.SGD(network2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # optimizrにMomentumSGDを指定
optKind2 = 'MomentumSGD'

network3 = CNNNet().to(device) # networkにさっき定義したnetworkを代入
optimizer3 = optim.Adam(network3.parameters(), lr=0.001) # optimizrにAdamを指定
optKind3 = 'Adam'

network4 = CNNNet().to(device) # networkにさっき定義したnetworkを代入
optimizer4 = optim.RMSprop(network4.parameters(), lr=0.0005, eps=1e-06) # optimizrにRMSpropを指定
optKind4 = 'RMSprop'


times, test_acc_list, train_loss_lists, train_acc_lists, val_loss_lists, val_acc_lists = [], [], [], [], [], []
Optims = [optKind1, optKind2, optKind3, optKind4]
Optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
networks = [network1, network2, network3, network4]

for i in range(len(Optims)):
    network = networks[i]
    optKind = Optims[i]
    optimizer = Optimizers[i]

    start_time = time.time()

    ###  training
    print('{} training start ...'.format(optKind))                      # 直す
    num_epochs = 50

    # initialize list for plot graph after training
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

    for epoch in range(num_epochs):
        # エポックごとに初期化
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        # ======== train_mode ======
        # 学習する
        network.train()
        for i, (images, labels) in enumerate(train_loader):  # ミニバッチ回数実行
            images, labels = images.to(device), labels.to(device) # そのまま使う
            optimizer.zero_grad()  # 勾配リセット
            outputs = network(images)  # 順伝播の計算
            loss = criterion(outputs, labels)  # lossの計算
            train_loss += loss.item()  # train_loss に結果を蓄積
            acc = (outputs.max(1)[1] == labels).sum()  # 予測とラベルが合っている数の合計
            train_acc += acc.item()  # train_acc に結果を蓄積
            loss.backward()  # 逆伝播の計算
            optimizer.step()  # 重みの更新
        avg_train_loss = train_loss / len(train_loader.dataset)  # lossの平均を計算
        avg_train_acc = train_acc / len(train_loader.dataset)  # accの平均を計算

        # ======== valid_mode ======
        # 評価する
        network.eval()
        with torch.no_grad():  # 必要のない計算を停止
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)  # そのまま使う
                outputs = network(images) # 出力を計算(順伝搬)
                loss = criterion(outputs, labels) # lossを計算
                val_loss += loss.item() # lossを足す
                acc = (outputs.max(1)[1] == labels).sum() # 正解のものを足し合わせてaccを計算
                val_acc += acc.item() # accを足す
        avg_val_loss = val_loss / len(valid_loader.dataset)  # lossの平均を計算
        avg_val_acc = val_acc / len(valid_loader.dataset)  # accの平均を計算

        # print log
        print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
              .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

        # append list for polt graph after training
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    stop_time = time.time()
    times.append(stop_time - start_time)

    # ======== fainal test ======
    # 最終評価する
    network.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # そのまま使う
            outputs = network(images)
            test_acc += (outputs.max(1)[1] == labels).sum().item()
            total += labels.size(0)
        print('test_accuracy: {} %'.format(100*test_acc / total))
        test_acc_list.append(100*test_acc / total)
        print('learning time: {:.3f} [sec]'.format(stop_time - start_time))


    # save weights
    torch.save(network.state_dict(), 'cnn_net_{}.ckpt'.format(optKind))

    train_loss_lists.append(train_loss_list)
    val_loss_lists.append(val_loss_list)
    train_acc_lists.append(train_acc_list)
    val_acc_lists.append(val_acc_list)



# plot graph
# グラフを用意して画像保存
import matplotlib.pyplot as plt

Color = ["c", "b", "r", "g"]
# save loss figure
plt.figure()
for i in range(len(Optims)):
    plt.plot(range(num_epochs), train_loss_lists[i], color=Color[i], linestyle='-',
             label='train_loss_{}'.format(Optims[i]))
    plt.plot(range(num_epochs), val_loss_lists[i], color=Color[i], linestyle='--',
             label='val_loss_{}'.format(Optims[i]))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('CNN Training and validation loss by various optimizers')
plt.grid()
plt.savefig('loss{}_by_{}.png'.format('CNN', 'various_optimizers'))

# save acc figure
plt.figure()
for i in range(len(Optims)):
    plt.plot(range(num_epochs), train_acc_lists[i], color=Color[i], linestyle='-',
             label='train_loss_{}'.format(Optims[i]))
    plt.plot(range(num_epochs), val_acc_lists[i], color=Color[i], linestyle='--',
             label='val_loss_{}'.format(Optims[i]))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('CNN Training and validation acc by various optimizers')
plt.grid()
plt.savefig('acc{}_by_{}.png'.format('CNN', 'various_optimizers'))

print('')
print('--------------------------------------------------------------')
print('')
for i in range(len(Optims)):
    print('Optimizer: {}  |  test_accuracy: {} %  |  learning time: {:.3f} [sec]'.format(Optims[i], test_acc_list[i], times[i]))