"""
FGSM against one model
white box attacks, include FGSM, DeepFool and PGD, for generating adversarial examples and
specifically customized for websites network taffic 1D dataset.

customized parameter alpha = 10 added , and is for the traffic trace in burst level, users can change it
based on your requirements

"""

import os
os.sys.path.append('..')


import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import copy, sys
from tqdm import tqdm
import torch.nn.functional as F
from src import util




class FGSM(object):

    def __init__(self,payload_size=1000,mode=None,x_box_min=0,x_box_max=256,pert_box=0.3,model=None, epsilon=0.1,first_n_byte=200000,num_append=500):
        """
        num_append: the length of bytes append at the end of each sample
        """
        self.payload_size = payload_size
        self.perts = np.random.randint(1,257,self.payload_size)  # number [1,256] exclude 257; 0 used as padding symbol when preprocessing
        self.model = model
        self.epsilon = epsilon
        self.mode = mode
        self.pert_box = pert_box
        self.x_box_min, self.x_box_max = x_box_min,x_box_max
        self.first_n_byte = first_n_byte
        self.num_append = num_append
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def reconstruction(self, x, y):
        """
        reconstruction restore original bytes from embedding matrix.

        Args:
            x torch.Tensor:
                x is word embedding

            y torch.Tensor:
                y is embedding matrix

        Returns:
            torch.Tensor:
        """
        x_size = x.size()[0]
        y_size = y.size()[0]
        # print(x_size, y_size)

        z = torch.zeros(x_size)

        for i in tqdm(range(x_size)):
            dist = torch.zeros(257)

            for j in range(y_size):
                dist[j] = torch.dist(x[i], y[j])  # computation of euclidean distance

            z[i] = dist.argmin()

        return z

    def perturbation(self, X_i, y):
        """
        given examples (X,y), returns corresponding adversarial example
        y should be a integer
        """

        X = np.copy(X_i.cpu())  #the input X_i is a tensor, so before copy, need to copy it in cpu; array
        X = torch.from_numpy(X)

        X_var = Variable(X.float(), requires_grad=True).to(self.device)
        y_var = Variable(y).to(self.device)

        # Creat embedding matrix
        embed = self.model.embed
        m = embed(torch.arange(0, 257).to(self.device)) # create lookup table [0,256],exclude 257

        embed_x = embed(X_var.long()).detach()
        embed_x.requires_grad = True
        output = self.model(embed_x)
        results = F.softmax(output, dim=1)
        loss = self.criterion(results, y_var)

        loss.backward(retain_graph=True)
        grad = embed_x.grad
        grad_sign = grad.detach().sign()[0][-self.payload_size:]
        pert = self.epsilon*grad_sign

        "embed intial pert"
        perts_init = embed(torch.from_numpy(self.perts).long().to(self.device))
        "normalize inital embed pert to [0,1]"
        normalizer = util.data_normalize_inverse(perts_init)
        perts_norm = normalizer.data_normalize()
        "update perturbation"
        pert = (perts_norm + pert.data.cpu().numpy())
        "inverse normalization"
        pert = normalizer.inverse_normalize(pert)

        "map pert to 0-256"
        pert = self.reconstruction(torch.Tensor(pert).to(self.device), m).detach().numpy()
        "clamp pert to 0-256"
        pert = torch.clamp(torch.tensor(pert).to(self.device),0,256).detach().cpu().numpy()
        X_sqz = X.squeeze()
        adv_x = np.concatenate([X_sqz.detach().numpy()[:-self.payload_size], pert.astype(np.uint8)])
        # add dimension
        adv_x = adv_x[np.newaxis, :]

        "get label for generated adv example"
        adv_x = Variable(torch.Tensor(adv_x)).to(self.device)
        embed_adv_x = embed(adv_x.long().to(self.device)).detach()
        adv_pred = self.model(embed_adv_x.to(self.device))
        adv_y = torch.argmax(adv_pred,1)

        return adv_y,adv_x,pert    # X in numpy format


class PGD(object):
    """
    iterative FGSM == PGD
    Optimizer is used to update weights of model based on loss.backward,
    and should not used in adversarial attacks,
    since the goal is to produce adversatial example based on target model,
    not to train the model,
    """

    def __init__(self,payload_size=1000,num_loop=10,x_box_min=0,x_box_max=256,pert_box=0.3,model=None, epsilon=0.1,first_n_byte=200000,num_append=500):
        """
        num_append: the length of bytes append at the end of each sample
        """
        self.payload_size = payload_size
        self.num_loop = num_loop
        self.model = model
        self.epsilon = epsilon
        self.pert_box = pert_box
        self.x_box_min, self.x_box_max = x_box_min,x_box_max
        self.first_n_byte = first_n_byte
        self.num_append = num_append
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def reconstruction(self, x, y):
        """
        reconstruction restore original bytes from embedding matrix.

        Args:
            x torch.Tensor:
                x is word embedding

            y torch.Tensor:
                y is embedding matrix

        Returns:
            torch.Tensor:
        """
        x_size = x.size()[0]
        y_size = y.size()[0]
        # print(x_size, y_size)

        z = torch.zeros(x_size)

        for i in tqdm(range(x_size)):
            dist = torch.zeros(257) # 257 zeros

            for j in range(y_size):
                dist[j] = torch.dist(x[i], y[j])  # computation of euclidean distance

            z[i] = dist.argmin()

        return z

    def perturbation(self, X_i, y):
        """
        given examples (X,y), returns corresponding adversarial example
        y should be a integer
        """
        # when preprocess the input, use 0 as padding symbol, the input from [1,256]
        pert = np.random.randint(1, 257, self.payload_size) # number from 1 to 256, exclude 257; keep consistent with preprocessing step
        X = np.copy(X_i.cpu()).squeeze()

        # get the embed layer
        embed = self.model.embed
        m = embed(torch.arange(0, 257).to(self.device)) # number from 0 to 256, exclude 257

        for i in range(self.num_loop):
            print(f'{i+1} time PGD attack')

            # self.optimizer.zero_grad()

            # inp_x = torch.from_numpy(np.concatenate([X[:-self.payload_size],pert])[np.newaxis,:]).float()
            inp_x = torch.from_numpy(np.concatenate([X[:-self.payload_size],pert]))
            inp_x = torch.reshape(inp_x,X_i.shape)
            inp_adv = inp_x.requires_grad_()
            embed_x = embed(inp_adv.long().to(self.device))
            embed_x.requires_grad = True

            output = self.model(embed_x)
            results = F.softmax(output, dim=1)

            r = results.cpu().detach().numpy()[0]
            print(f'Acc as Benign: {r[0]}, Acc as Malware: {r[1]}')

            # make a decision when to terminate
            if r[0] > 0.5:
                print(f'PGD attacking rate: {r[0]}')
                break

            # get loss
            loss = self.criterion(results, y.to(self.device))
            print(f'Loss: {loss.item()}')

            loss.backward()
            # self.optimizer.step()

            # get sign
            grad = embed_x.grad
            grad_sign = grad.detach().sign()[0][-self.payload_size:]

            # get perturbation
            pert = embed(torch.from_numpy(pert).long().to(self.device))
            "normalize inital embed pert to [0,1]"
            normalizer = util.data_normalize_inverse(pert)
            perts_norm = normalizer.data_normalize()
            "update perturbation"
            pert = (perts_norm - (self.epsilon*grad_sign).data.cpu().numpy())
            "inverse normalization"
            pert = normalizer.inverse_normalize(pert)
            "convert ndarray to tensor"
            pert = torch.Tensor(pert).to(self.device)
            "map to original space via embed layer"
            pert = self.reconstruction(pert, m).detach().numpy()
            "clamp pert within 0-256"
            pert = torch.clamp(pert,0,256)

        adv_y = torch.argmax(results,1)
        adv_x = inp_x
        return adv_y, adv_x, pert    # X in numpy format
