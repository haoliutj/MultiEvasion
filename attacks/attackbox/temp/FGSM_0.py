import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import numpy as np

from attack_base import Attack
from src.util import reconstruction

class FGSM(Attack):

    """
    FSGM based on TWO Models:
        - FGSM in the paper 'Explaining and harnessing adversarial examples'
        - [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf or others

    Arguments:
        model_1 (nn.Module): model_1 to attack. (default as Malconv)
        model_2: model_2 to attack. (default as DNN from FireEye)
        index_to_perturb: a list of index can be added noise;
            - Default as first part of DOS header [2,0x3c] ([2,60]), except the first Magic number "MZ"
        eps (float): maximum perturbation. (DEFALUT: 0.007)
        random_init:
            - True --> add random (Gaussim) noise to input x at first place to improve the performance
            - False --> no changes to input x
        pert_init_with_benign:
            - True --> initiate perturbations by adding bytes from benign files (randomly select) to the position of index_to_perturb
            - False --> random initiate values in position of index_to_perturb to value range in [0,256]
        w_1: weight to control the weight of model_1 when compute the loss function
        w_2: weight to control the weight of model_2 when compute the loss function


    Shape:
        - inputs: `(N, W)` where `N = number of batches`,  `W = width`.
        - labels: `(N)` where `N = number of batches`, and each value is the corresponding label
        - output: :math:`(N, W)`.

    Examples:
        > attack = attackbox.FGSM(model, eps=0.007,index_to_perturb=l)
        > adv_Xs = attack(inputs, labels)

    """

    def __init__(self,
                 model_1,
                 model_2 = None,
                 eps: float=0.07,
                 w_1: float=0.5,
                 w_2: float=0.5,
                 index_to_perturb:list=None,
                 random_init:bool=True,
                 pert_init_with_benign:bool=True):

        super(FGSM,self).__init__("FGSM",model_1,model_2)

        if not index_to_perturb:
            index_to_perturb = [i for i in range(2,0x3c)]
            print("Crafting based on partial DOS header")

        self.w_1, self.w_2 = w_1, w_2
        self.eps = eps
        self.index_to_perturb = index_to_perturb
        self.random_init = random_init
        self.pert_init_with_benign = pert_init_with_benign
        self.loss = nn.CrossEntropyLoss()


    def perturbation(self,inputs,labels):

        X = inputs.clone().detach.to(self.device)
        y = labels.clone().detach.to(self.device)

        if self.random_init:
            X = X + torch.Tensor(np.random.uniform(-self.eps,self.eps,X.shape)).type_as(X).to(self.device)
        X = Variable(X.float(), requires_grad=True).to(self.device)

        # obtain embedding layer and create embedding matrix
        embed_1 = self.model_1.embed
        embed_matrix = embed_1(torch.arange(0,257).to(self.device))

        # forward process

        embed_1_x = embed_1(X.long()).detach()

        embed_1_x.requires_grad = True

        output_1 = self.model_1(embed_1_x)

        output_1 = F.softmax(output_1,dim=1)

        loss_1 = self.targeted * self.loss(output_1, self.targeted_label)

        embed_2 = self.model_2.embed
        embed_2_x = embed_2(X.long()).detach()
        embed_2_x.requires_grad = True
        output_2 = self.model_2(embed_2_x)
        output_2 = F.softmax(output_2, dim=1)
        loss_2 = self.targeted * self.loss(output_2, self.targeted_label)

        loss = self.w_1 * loss_1 + self.w_2 * loss_2

        loss.backward()



