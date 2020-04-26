import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import matplotlib.image as img
import time

# load MNIST data
train_Dataset = dsets.CIFAR10(  # CIFAR10 default dataset
    root='./data_cifar10/',  # rootで指定したフォルダーを作成して生データを展開。これは必須。
    train=True,  # 学習かテストかの選択。これは学習用のデータセット
    transform=transforms.ToTensor(),  # Pytroch のテンソルに変換する  # 自動で形式を変換するのでラベルをOne_hotにする必要等がない。
    download=True)  # ダウンロードするかどうかの選択(当然する) ここには正解ラベルのデータと入力文字のデータの対がまとめて入っている。.

test_dataset = dsets.CIFAR10(
    root='./data_cifar10/',
    train=False, # 学習かテストかの選択。これはテスト用のデータセット
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

# Multi Layer Perceptron Network
# 多層ニューラルネットワークの構築。多層にする。今回は3層。
# (Affine relu (dropout) | Affine relu (dropout) | Affine relu (Softmax))
# nn.Moduleを継承している。
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # xAT + b (重みとバイアスを足し合わせる作業と、重みパラメータを保持する機能)
        self.fc1 = nn.Linear(32*32*3, 1024) # 3072の配列を1024に線形変換(入力) (in, out)
        nn.init.kaiming_normal_(self.fc1.weight) # reluなので、Heの初期値に重みを初期化
        self.fc2 = nn.Linear(1024, 1024) # 1024の配列を1024に線形変換
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(1024, 10) # 1024の配列を10に線形変換(出力)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x): # predictに相当(順伝搬)
        x = F.relu(self.fc1(x)) # １つ１つの活性化関数にbackwardの定義ある。
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))

    # accはoptimizerを用いた学習を記述する中で計算をする。(1エポックごとに)

    # 損失計算の定義は必要ない。(optimizingのcriterionでloss関数に交差エントロピーを指定)

    # 勾配計算の定義も必要ない。(optimizingで指定するcriterionがbackwardを計算してくれる)

# select device
# GPU or CPU? (利用可能ならGPUを使う)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = MLPNet().to(device) # networkにさっき定義したnetworkを代入

# optimizing
criterion = nn.CrossEntropyLoss() # 交差エントロピーをloss計算に用いる。
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # optimizrに勾配確率降下法を指定
optKind = 'SGD'

start_time = time.time()

###  training
print('training start ...')
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
        # viewで32*32*3画像を１次元に変換し、deviceへ転送
        images, labels = images.view(-1, 32*32*3).to(device), labels.to(device)
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
            # viewで32*32*3画像を１次元に変換し、deviceへ転送
            images, labels = images.view(-1, 32*32*3).to(device), labels.to(device)
            outputs = network(images) # 出力を計算(順伝搬)
            loss = criterion(outputs, labels) # lossを計算
            val_loss += loss.item() # lossを足す
            acc = (outputs.max(1)[1] == labels).sum() # 正解のものを足し合わせてaccを計算
            val_acc += acc.item() # accを足す
    avg_val_loss = val_loss / len(valid_loader.dataset)  # lossの平均を計算
    avg_val_acc = val_acc / len(valid_loader.dataset)  # accの平均を計算

    # print log
    print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
          .format(epoch + 1, num_epochs, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

    # append list for polt graph after training
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

stop_time = time.time()

# ======== fainal test ======
# 最終評価する
network.eval()
with torch.no_grad():
    total = 0
    test_acc = 0
    for images, labels in test_loader:
        images, labels = images.view(-1, 32*32*3).to(device), labels.to(device)
        outputs = network(images)
        test_acc += (outputs.max(1)[1] == labels).sum().item()
        total += labels.size(0)
    print('test_accuracy: {} %'.format(100 * test_acc / total))
    print('learning time: {:.3f} [sec]'.format(stop_time - start_time))

# save weights
torch.save(network.state_dict(), 'mnist_net.ckpt')

# plot graph
# グラフを用意して画像保存
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('MLP Training and validation loss by {} (Few Layers)'.format(optKind))
plt.grid()
plt.savefig('loss{}.png'.format('MLP_fewLayers'))

plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('MLP Training and validation accuracy by {} (Few Layers)'.format(optKind))
plt.grid()
plt.savefig('acc{}.png'.format('MLP_fewLayers'))