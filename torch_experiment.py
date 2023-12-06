import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from models import *
from utils import progress_bar
import copy
import torch.nn.init as init
import math
 
import model


n = 256
bn = False
bs = 50
Tmax = 500

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default= 0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def normalize_data(x):
    for i in range(x.size()[0]):
        x[i] = x[i] - torch.mean(x[i])
        x[i] = x[i]/torch.norm(x[i])
    return x

#def normalize_data(x):
#    return x

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.1307, ), (0.3081, )),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.1307, ), (0.3081, )),
])


trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_train)


C = 1
Nall = 5000

def create_subset(a,b, dataset):
    places = []
    targets = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == a:
            places.append(i)
            targets.append( -1)
        if  dataset.targets[i] == b:
            places.append(i)
            targets.append( 1)
    dataset.data = dataset.data[places,:,:]
    dataset.targets = targets
    return dataset


trainset = create_subset(2,3,trainset)
testset = create_subset(2,3,testset)





#trainset.data = trainset.data[0:100]
#trainset.data = trainset.targets[0:100]

#a = (trainset.targets == 1)
#b = (trainset.targets == 2)
#c = a + b
#trainset.data = trainset.data[c]
#trainset.targets = trainset.targets[c]



trainset.data = trainset.data[0:Nall]
trainset.targets = trainset.targets[0:Nall]

testset.data = testset.data[0:Nall]
testset.targets = testset.targets[0:Nall]


#trainset.data = trainset.data[p]
#trainset.targets = trainset.targets[p]


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=1)

trainloader2 = torch.utils.data.DataLoader(
    trainset, batch_size=bs, shuffle=False, num_workers=1)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



class Net(nn.Module):
    def __init__(self, n , bn,c):
        super(Net, self).__init__()
        self.n = n
        self.c = c
        self.bias = True
        self.fc1 = nn.Linear(28*28, n, bias = self.bias)
        self.fc2 = nn.Linear(n, n, bias = self.bias)
        self.fc3 = nn.Linear(n, self.c, bias = self.bias)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class Net1(nn.Module):
    def __init__(self, n , bn,c):
        super(Net1, self).__init__()
        self.n = n
        self.c = c
        self.bias = False
        self.fc1 = nn.Linear(28*28, n, bias = self.bias)
        self.fc2 = nn.Linear(n, n, bias = self.bias)
        #self.fc3 = nn.Linear(n, self.c, bias = self.bias)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.fc3(x)/math.sqrt(self.n)
        return x

class Net2(nn.Module):
    def __init__(self, n , bn,c):
        super(Net2, self).__init__()
        self.n = n
        self.c = c
        self.bias = False
        #self.fc1 = nn.Linear(28*28, n, bias = self.bias)
        #self.fc2 = nn.Linear(n, n, bias = self.bias)
        self.fc3 = nn.Linear(n, self.c, bias = self.bias)

    def forward(self, x):
        x = self.fc3(x)
        return x



    
# Model
print('==> Building model..')
net = Net(n,bn,C)
net = net.to(device)
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay= 0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)


def extract_prams(net):
    grads = []
    sizes = []
    for param in net.parameters():
        sizes.append(param.size())
        grads.append(param.view(-1))
    return torch.cat(grads)

def extract_grads(net):
    grads = []
    sizes = []
    for param in net.parameters():
        sizes.append(param.size())
        grads.append(param.grad.view(-1))
    return grads,sizes


def extract_initial_output(net):
    net.eval()
    out = torch.zeros((Nall,C))
    for batch_idx, (inputs, targets) in enumerate(trainloader2):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(normalize_data(inputs))
        out[ batch_idx*bs:(batch_idx+1)*bs] = outputs
    return out



def extract_features(net, x, total_params, optimizer):
    num_data = x.size()[0]
    net.train()
    K = torch.zeros((num_data,total_params, C))
    for i in range(num_data):
        for j in range(10):
            optimizer.zero_grad()
            net(x)[i,j].backward()
            jac1 = []
            for param in net.parameters():
                jac1.append(param.grad.view(-1))
            jac1 = torch.cat(jac1)
            K[i,:,j] = jac1
            print(jac1)
    return K



def construct_temp_net(initial_param, final_param, t, bn):
    temp_param = []
    temp_net = Net(n,bn,C)
    temp_net = net.to(device)
    for i in range(len(initial_param)):
        temp_param.append( (1-t)*torch.tensor(initial_param[i]).detach() + t*torch.tensor(initial_param[i]).detach())
    i = 0
    with torch.no_grad():
        for param in temp_net.parameters():
            param.copy_(temp_param[i].clone().detach())
            i = i+1
    temp_optimizer = optim.SGD(temp_net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-5)
    # updating running stats
    if bn:
        temp_net.train()
        for i in range(5):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = temp_input(inputs)
    return temp_net, temp_optimizer



# Training
def train(epoch, optimizer, net, theta0 ):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #y = torch.nn.functional.one_hot(targets, num_classes=C)
        #y = targets.float()
        y = torch.sign(torch.squeeze(net2(net1(normalize_data(inputs)).detach())).detach().float())
        optimizer.zero_grad()
        outputs = torch.squeeze(net(normalize_data(inputs)))
        thetat = extract_prams(net)
        loss = criterion(outputs, y) + 0*criterion(thetat, theta0)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.squeeze(outputs.sign())
        total += targets.size(0)
        correct += predicted.eq(y).sum().item()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total, train_loss/(batch_idx+1)


def train2(epoch, optimizer, net1,net2):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = targets.float()
        optimizer.zero_grad()
        outputs = torch.squeeze(net2(net1(normalize_data(inputs)).detach()))
        thetat = extract_prams(net2)
        loss = criterion(outputs, y) + 0*criterion(thetat, theta0)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.squeeze(outputs.sign())
        total += targets.size(0)
        correct += predicted.eq(y).sum().item()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total, train_loss/(batch_idx+1)



def get_error(temp_optimizer, temp_net):
    temp_net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            y = targets.float()
            outputs = torch.squeeze(temp_net(normalize_data(inputs)))
            loss = criterion(outputs, y)
            test_loss += loss.item()
    return test_loss/(batch_idx+1)



total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight, mean=0.0, std= math.sqrt(
                1/(m.in_channels * m.kernel_size[0] * m.kernel_size[1])))
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0.0, std= math.sqrt(2/m.in_features))
            


#########

net1 = Net1(n,bn,C)
net1 = net1.to(device)
net2 = Net2(n,bn,C)
net2 = net2.to(device)

optimizer2 = optim.SGD(net2.parameters(), lr= 0.1,
                      momentum=0.9, weight_decay= 0)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

init_params(net1)
init_params(net2)

epoch = 0
acc = 0
theta0 = extract_prams(net2).detach()


L = []

while acc < 101 and epoch < 400:
    acc, loss_train = train2(epoch, optimizer2, net1, net2 )
    #test(epoch, optimizer, net )
    scheduler2.step()
    print(acc, loss_train)
    L.append(loss_train)
    epoch +=1
plt.plot(L)
plt.show()





########





            

init_params(net)
initial_net = copy.deepcopy(net)

initial_param = list(initial_net.parameters())

ini_out = extract_initial_output(net)
ini_out = ini_out.detach().to(device)


acc = 0
epoch = 0
ACC = []
L = []

theta0 = extract_prams(initial_net).detach()

while acc < 101 and epoch < Tmax:
    acc, loss_train = train(epoch, optimizer, net, theta0 )
    #test(epoch, optimizer, net )
    scheduler.step()
    print(acc, loss_train)
    L.append(loss_train)
    epoch +=1
plt.plot(L)
plt.show()
    


def mody(x):
    y = torch.sign(torch.squeeze(net2(net1(normalize_data(x)).detach())).detach().float())
    return y

    
def construct_temp_net(initial_param, final_param, t, bn):
    temp_param = []
    temp_net = Net(n,bn,C)
    temp_net = temp_net.to(device)
    for i in range(len(initial_param)):
        temp_param.append(  (1-t)*torch.tensor(initial_param[i].detach() ) +   (t)*torch.tensor(final_param[i].detach() )  )
    i = 0
    with torch.no_grad():
        for param in temp_net.parameters():
            param.copy_(temp_param[i])
            i = i+1
    temp_optimizer = optim.SGD(temp_net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-5)
    # updating running stats
    if bn:
        temp_net.train()
        for i in range(5):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = temp_net(inputs)
    return temp_net, temp_optimizer


final_param = list(net.parameters())


######## network error ###########
error = []
ps = 50
sample_point = np.linspace(0,1,ps)
for t in sample_point:
    print(t)
    temp_net, temp_optimizer = construct_temp_net(initial_param, final_param, t, bn)
    error.append(get_error(temp_optimizer, temp_net))
plt.plot(sample_point,error)
plt.xlabel('t')
plt.ylabel('train loss')
plt.show()
    
####### nabla bar #######


ip = torch.cat([ j.detach().view(-1) for j in initial_param])
fp = torch.cat([ j.detach().view(-1) for j in final_param])


def extract_features(temp_net, x, total_params, temp_optimizer):
    num_data = x.size()[0]
    temp_net = temp_net.to(device)
    temp_net.train()
    K = torch.zeros((num_data,total_params, C))
    for i in range(num_data):
        for j in range(C):
            temp_optimizer.zero_grad()
            loss = temp_net(  x  )[i,j]
            loss.backward()
            jac1 = []
            for param in temp_net.parameters():
                jac1.append(param.grad.view(-1))
            jac1 = torch.cat(jac1)
            K[i,:,j] = jac1
    return K


maxnum = 500

final_out = extract_initial_output(net)
final_out = final_out.detach().to(device)
final_out = final_out[0:maxnum,:]
final_out = final_out.cpu()

diff_param = fp - ip




es_out = torch.zeros((ps,maxnum,C))

fbar = torch.zeros((maxnum, total_params ,C))

for out_iter in range(len(sample_point)):
    print(out_iter)
    t = sample_point[out_iter]
    #es_out = torch.zeros((maxnum,10))
    #es_out = es_out.to(device)
    temp_net, temp_optimizer = construct_temp_net(initial_param, final_param, t, bn)
    for batch_idx, (inputs, targets) in enumerate(trainloader2):
        if bs*batch_idx < maxnum:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = normalize_data(inputs)
            a = extract_features(temp_net, inputs, total_params, temp_optimizer)
            fbar[batch_idx*bs:(batch_idx+1)*bs,:,:] += a/ps 
            a = a.to(device)
            temp_out = ini_out[bs*batch_idx: bs*(batch_idx+1), : ]
            for i in range(C):
                es_out[out_iter,batch_idx*bs:(batch_idx+1)*bs,i] = ini_out[batch_idx*bs:(batch_idx+1)*bs,i] + torch.squeeze(a[:,:,i])@diff_param



valuebar = torch.zeros((maxnum,C)).to(device)
fbar= fbar.to(device)


for i in range(C):
    valuebar[:,i] = ini_out[0:maxnum,i] + torch.squeeze(fbar[:,:,i])@diff_param

valuebar = valuebar.cpu()


out_diff = torch.zeros((ps,maxnum,C))


for i in range(ps):
    out_diff[i,:,:] = torch.abs(es_out[i,:,:] - final_out)
    

inds = torch.zeros((maxnum,C))
values = torch.zeros((maxnum,C))



for i in range(maxnum):
    for j in range(C):
        temp = out_diff[:,i,j]
        k = torch.argmin(temp)
        inds[i,j] = k
        values[i,j] = es_out[k,i,j]
        

maxsnr = 20*torch.log10(torch.norm(final_out)/torch.norm(values - final_out))
maxsnr2 = 20*torch.log10(torch.norm(final_out)/torch.norm(valuebar - final_out))
snr = []
for i in range(ps):
    snr.append( 20*torch.log10(torch.norm(final_out)/torch.norm(es_out[i,:,:] - final_out)) )
plt.plot(sample_point, snr)
plt.xlabel('t')
plt.ylabel('snr')
plt.plot(sample_point, [maxsnr]*ps)
plt.plot(sample_point, [maxsnr2]*ps)
plt.show()


plt.hist(inds.view(-1)/ps,ps)
plt.show()


####### constructing NTK matrix ##########

def extract_features_2(model, x, total_params, op, num_out):
    num_data = x.size()[0]
    model = model.to(device)
    model.eval()
    K = torch.zeros((num_data,total_params))
    for i in range(num_data):
        op.zero_grad()
        loss = model(x)[i,num_out]
        loss.backward()
        jac1 = []
        for param in model.parameters():
            jac1.append(param.grad.view(-1))
        jac1 = torch.cat(jac1)
        K[i,:] = jac1
    return K




N = maxnum




x = torch.zeros((Nall,1,28,28)).to(device)
y = torch.zeros((Nall,C))
allt = inds/ps

for batch_idx, (inputs, targets) in enumerate(trainloader2):
    inputs, targets = inputs.to(device), targets.to(device)
    y[batch_idx*bs:(batch_idx+1)*bs,:] = mody(inputs)[:,None]
    inputs = normalize_data(inputs)
    x[batch_idx*bs:(batch_idx+1)*bs] = inputs
    
x = x[0:N]
y = y[0:N]


K0 = torch.zeros((N,N,C))

for h in range(C):
    print(h)
    for i in range(N):
        temp_net_1, temp_optimizer_1 = construct_temp_net(initial_param, final_param, 0, bn)
        f1 = extract_features_2(temp_net_1, x, total_params, temp_optimizer_1,h)
        f1 = torch.squeeze(f1)
        K0[ :, :,h ] = torch.inner(f1,f1)



Kf = torch.zeros((N,N,C))

for h in range(C):
    print(h)
    for i in range(N):
        temp_net_1, temp_optimizer_1 = construct_temp_net(initial_param, final_param, 1, bn)
        f1 = extract_features_2(temp_net_1, x, total_params, temp_optimizer_1,h)
        f1 = torch.squeeze(f1)
        Kf[ :, :, h ] = torch.inner(f1,f1)
                
                
Kbar = torch.zeros((N,N,C))

for h in range(C):
    print(h)
    f1 = torch.squeeze(fbar[:,:,h])
    Kbar[ :, :,h ] = torch.inner(f1,f1)



def nk(K):
    K = K - torch.min(K)
    K = K/torch.max(K)
    return K




net.eval()

with torch.no_grad():
    y = net(x)

with torch.no_grad():
    y0 = initial_net(x)

Ky = torch.zeros((N,N,C))


for h in range(C):
    for i in range(N):
        for j in range(N):
            Ky[i,j,h] = (y[i,h] - y0[i,h])*(y[j,h] - y0[j,h])



plt.imshow(torch.squeeze(nk(K0[0:50,0:50,0])))
plt.title('K0')
plt.show()
plt.imshow(torch.squeeze(nk(Kf[0:50,0:50,0])))
plt.title('Kf')
plt.show()
plt.imshow(torch.squeeze(nk(Kbar[0:50,0:50,0] )))
plt.title('Kbar')
plt.show()
plt.imshow(Ky[0:50,0:50,0])
plt.title('Ky')
plt.show()





vk0 = K0[:,:,0].reshape(-1)
vkf = Kf[:,:,0].reshape(-1)
vky = Ky[:,:,0].reshape(-1)
vkbar = Kbar[:,:,0].reshape(-1)


def cos(a,b):
    c = torch.inner(a,b)
    c = c/torch.norm(a)
    c = c/torch.norm(b)
    return c


tt = np.linspace(1, 3, 100)
s1 = []
for i in tt:
    temp = vkbar - i*vk0
    s1.append( cos(temp,vky) )
plt.plot(tt,s1)

tt = np.linspace(1, 3, 100)
s2 = []
for i in tt:
    temp = vkf - i*vk0
    s2.append( cos(temp,vky) )
plt.plot(tt,s2)
plt.show()





M = torch.zeros((N**2,2))
b = torch.zeros((N**2,1))
M[:,0] = vk0
M[:,1] = vky
b = vkbar


q = torch.linalg.pinv(M)
w = q@b




K1 = Kbar - w[0]*K0
K2 = K1 - w[1]*Ky

plt.imshow(nk(K2[0:50,0:50,0]))
plt.show()

a = torch.real(torch.linalg.eig(K2[:,:,0])[0])

plt.plot(a[0:100])
plt.show()


from sklearn.decomposition import PCA

a = PCA(2)


xx = x.reshape(500,-1)
xx = xx.detach().cpu().numpy()

b = a.fit_transform(xx)

g = torch.diagonal(K2)
z = torch.argsort(g,descending = True)[0]
y_real = y.cpu()

y_hat = net(x).detach().cpu()

y_hat2 = mody(x).detach().cpu()




plt.scatter(b[torch.where(y_hat < 0)[0],1],b[torch.where(y_hat < 0)[0],0], c = 'blue')
plt.scatter(b[torch.where(y_hat > 0)[0],1],b[torch.where(y_hat > 0)[0],0], c = 'red')
plt.scatter(b[z[1:20],1],b[z[1:20],0], c = 'green')
plt.show()




def mody2(x):
    y = torch.squeeze(net2(net1(normalize_data(x)).detach())).detach().float()
    return y

gx = mody2(x).detach().cpu()
gy = g.detach().cpu()

plt.scatter(gx,gy)



np.save('K0.npy',K0.cpu().numpy() )
np.save('Kfinal.npy',Kf.cpu().numpy() )
np.save('Kbar.npy',Kbar.cpu().numpy() )
np.save('Kyy.npy',Ky.cpu().numpy() )
np.save('data.npy', x.cpu().numpy())
np.save('yhat.npy', y.cpu().numpy())

np.save('Kssup.npy', K2.cpu().numpy())


np.save('diagonal_Ksuup.npy', gx.cpu().numpy())
np.save('smooth_out.npy', gy.cpu().numpy())

np.save('pca_components.npy', b)


