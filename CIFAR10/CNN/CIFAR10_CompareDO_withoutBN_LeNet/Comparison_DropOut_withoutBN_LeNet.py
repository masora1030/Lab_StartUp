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
# form = transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize(
#                 [0.5, 0.5, 0.5],  # RGB 平均
#                 [0.5, 0.5, 0.5]  # RGB 標準偏差
#            )
#        ])

# load CIFAR10 data
train_Dataset = dsets.CIFAR10(  # CIFAR10 default dataset
    root='./data_cifar10/',  # rootで指定したフォルダーを作成して生データを展開。これは必須。
    train=True,  # 学習かテストかの選択。これは学習用のデータセット
    # transform=form,  # 正規化
    transform=transforms.ToTensor(),  # Pytroch のテンソルに変換する  # 自動で形式を変換するのでラベルをOne_hotにする必要等がない。
    download=True)  # ダウンロードするかどうかの選択(当然する) ここには正解ラベルのデータと入力文字のデータの対がまとめて入っている。.

test_dataset = dsets.CIFAR10(
    root='./data_cifar10/',
    train=False, # 学習かテストかの選択。これはテスト用のデータセット
    # transform=form,  # 正規化
    transform=transforms.ToTensor(),  # Pytroch のテンソルに変換する。 # 自動で形式を変換するのでラベルをOne_hotにする必要等がない。
    download=True)  # ダウンロードするかどうかの選択(当然する)

train_dataset, valid_dataset = torch.utils.data.random_split(  # データセットの分割(validation)
    train_Dataset,  # 分割するデータセット(さっきダウンロードしたやつ)
    [40000, 10000])  # 分割数    (50000個の学習用データを、さらに40000個の学習用データと10000個のバリデーション用データに分割)

print('train_dataset = ', len(train_dataset)) # 40000
print('valid_dataset = ', len(valid_dataset)) # 10000
print('test_dataset = ', len(test_dataset)) # 10000


# set data loader
# データセットからミニバッチ単位でデータを取り出し、ネットワークへ供給するためのローダー(バッチ処理の準備)
# バッチサイズを変更すると、勾配のノイズ(不安定差)が変わる。
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,  # データセットの指定
    batch_size=64,  # ミニバッチの指定
    shuffle=True,  # シャッフルするかどうかの指定
    num_workers=2)  # コアの数

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2)

class MyCNN1(torch.nn.Module):
    def __init__(self):
        super(MyCNN1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,  # チャネル入力
                                     6,  # チャンネル出力
                                     5,  # カーネルサイズ
                                     1,  # ストライド (デフォルトは1)
                                     0,  # パディング (デフォルトは0)
                                     )
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.pool = torch.nn.MaxPool2d(2, 2)  # カーネルサイズ, ストライド

        self.dropout1 = torch.nn.Dropout2d(p=0.3)  # [new] Dropoutを追加してみる

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 入力サイズ, 出力サイズ
        self.dropout2 = torch.nn.Dropout(p=0.5)  # [new] Dropoutを追加してみる
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = self.dropout1(x)  # [new] Dropoutを追加
        x = x.view(-1, 16 * 5 * 5)  # 1次元データに変えて全結合層へ
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)  # [new] Dropoutを追加
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class MyCNN2(torch.nn.Module):
    def __init__(self):
        super(MyCNN2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,  # チャネル入力
                                     6,  # チャンネル出力
                                     5,  # カーネルサイズ
                                     1,  # ストライド (デフォルトは1)
                                     0,  # パディング (デフォルトは0)
                                     )
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.pool = torch.nn.MaxPool2d(2, 2)  # カーネルサイズ, ストライド

        self.dropout1 = torch.nn.Dropout2d(p=0.3)  # [new] Dropoutを追加してみる

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 入力サイズ, 出力サイズ
        self.dropout2 = torch.nn.Dropout(p=0.5)  # [new] Dropoutを追加してみる
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)  # [new] Dropoutを追加
        x = x.view(-1, 16 * 5 * 5)  # 1次元データに変えて全結合層へ
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)  # [new] Dropoutを追加
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class MyCNN3(torch.nn.Module):
    def __init__(self):
        super(MyCNN3, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,  # チャネル入力
                                     6,  # チャンネル出力
                                     5,  # カーネルサイズ
                                     1,  # ストライド (デフォルトは1)
                                     0,  # パディング (デフォルトは0)
                                     )
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.pool = torch.nn.MaxPool2d(2, 2)  # カーネルサイズ, ストライド

        self.dropout1 = torch.nn.Dropout2d(p=0.3)  # [new] Dropoutを追加してみる

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 入力サイズ, 出力サイズ
        self.dropout2 = torch.nn.Dropout(p=0.5)  # [new] Dropoutを追加してみる
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)  # [new] Dropoutを追加
        x = x.view(-1, 16 * 5 * 5)  # 1次元データに変えて全結合層へ
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # [new] Dropoutを追加
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# select device
# GPU or CPU? (利用可能ならGPUを使う)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# optimizing
criterion = nn.CrossEntropyLoss() # 交差エントロピーをloss計算に用いる。




# ここから比較のコード


network1 = MyCNN1().to(device) # networkにさっき定義したnetworkを代入
optimizer1 = optim.SGD(network1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # optimizrにMomentumSGDを指定
optKind1 = 'Default(NoDropOut)'

network2 = MyCNN2().to(device) # networkにさっき定義したnetworkを代入
optimizer2 = optim.SGD(network2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # optimizrにMomentumSGDを指定
optKind2 = '1DropOut'

network3 = MyCNN3().to(device) # networkにさっき定義したnetworkを代入
optimizer3 = optim.SGD(network3.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # optimizrにMomentumSGDを指定
optKind3 = '2DropOuts'


times, test_acc_list, train_loss_lists, train_acc_lists, val_loss_lists, val_acc_lists = [], [], [], [], [], []
Optims = [optKind1, optKind2, optKind3]
Optimizers = [optimizer1, optimizer2, optimizer3]
networks = [network1, network2, network3]

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

for i in range(len(Optims)):
    plt.figure()
    plt.plot(range(num_epochs), train_loss_lists[i], color='blue', linestyle='-', label='train_loss')
    plt.plot(range(num_epochs), val_loss_lists[i], color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('CNN Training and validation loss with {}'.format(Optims[i]))
    plt.grid()
    plt.savefig('loss{}_with_{}.png'.format('VGG16', Optims[i]))

for i in range(len(Optims)):
    plt.figure()
    plt.plot(range(num_epochs), train_acc_lists[i], color='blue', linestyle='-', label='train_loss')
    plt.plot(range(num_epochs), val_acc_lists[i], color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('CNN Training and validation acc with {}'.format(Optims[i]))
    plt.grid()
    plt.savefig('acc{}_with_{}.png'.format('VGG16', Optims[i]))



Color = ["c", "b", "r", "g", "y", "m", "k", '#a65628']
# save loss figure
plt.figure().patch.set_facecolor('white')
for i in range(len(Optims)):
    plt.plot(range(num_epochs), train_loss_lists[i], color=Color[i], linestyle='-',
             label='train_{}'.format(Optims[i]))
    plt.plot(range(num_epochs), val_loss_lists[i], color=Color[i], linestyle='--',
             label='val_{}'.format(Optims[i]))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=13)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('CNN Training and validation loss with various Drop Out')
plt.grid()
plt.savefig('loss{}_by_{}.png'.format('CNN', 'various_Drop_Out'), bbox_inches="tight")

# save acc figure
plt.figure().patch.set_facecolor('white')
for i in range(len(Optims)):
    plt.plot(range(num_epochs), train_acc_lists[i], color=Color[i], linestyle='-',
             label='train_{}'.format(Optims[i]))
    plt.plot(range(num_epochs), val_acc_lists[i], color=Color[i], linestyle='--',
             label='val_{}'.format(Optims[i]))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=13)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('CNN Training and validation acc with various Drop Out')
plt.grid()
plt.savefig('acc{}_by_{}.png'.format('CNN', 'various_Drop_Out'), bbox_inches="tight")

print('')
print('--------------------------------------------------------------')
print('')
for i in range(len(Optims)):
    print('Batch Norms: {}  |  test_accuracy: {} %  |  learning time: {:.3f} [sec]'.format(Optims[i], test_acc_list[i], times[i]))
