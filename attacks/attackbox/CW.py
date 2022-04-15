"""
CW attack for models with raw bytes input
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from attacks.attackbox.attack_base import Attack
from src import util



class CW(Attack):
    """
    CW based on TWO Models: can only take one input sample each time
        - CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
        - [https://arxiv.org/abs/1608.04644]

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
        c (float): c in the paper. parameter for box-constraint. (Default: 1e-4)
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 1000)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

        warning:: With default c, you can't easily get adversarial images. Set higher c like 1.


    Shape:
        - inputs: `(N, W)` where `N = number of batches`,  `W = width`. N has to be 1 !!
        - labels: `(N)` where `N = number of batches`, and each value is the corresponding label
        - output: :math:`(N, W)`.

    return:
        - if one target model: prediction scores, label of adversarial malware, size of perturbation (i.e., length of pert)
        - if two target models: (prediction_scores_1, adv_y_1), (prediction_scores_2, adv_y_2),size of perturbation (i.e., length of pert)

    """

    def __init__(self,
                 model_1=None,
                 model_2=None,
                 c:float = 1e-4,
                 kappa=0,
                 iter_steps:int = 1000,
                 lr:float = 0.1,
                 w_1: float = 0.5,
                 w_2: float = 0.5,
                 index_to_perturb: list = None, # can also obtain in perturbation step
                 random_init: bool = True,
                 pert_init_with_benign: bool = True):

        ## 继承父类Attack的参数
        super(CW, self).__init__(model_1, model_2)
        ## 不同方式继承父类
        #Attack.__init__(model_1, model_2)

        if not index_to_perturb:
            index_to_perturb = [i for i in range(2, 0x3c)] ## partial dos length=58 ( magic numbers in DOS header "MZ"[0x00-0x02] and [0x3c-0x40] can't be modified)
            # print("Initiate crafting process with partial DOS header...")

        self.w_1, self.w_2 = w_1, w_2
        self.c = c
        self.kappa = kappa
        self.iter_steps = iter_steps
        self.lr = lr
        self.index_to_perturb = index_to_perturb
        self.random_init = random_init
        self.pert_init_with_benign = pert_init_with_benign
        self.attack = 'CW'

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        if self.targeted==1:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)


    ## replace the value that not in the list index_to_perturb with original value
    def validate_advX(self,x=None,adv_x=None,index_to_perturb=None):
        """
        replace the values of adv_x that the index not in the index_to_perturb with corresponding
        values in the x
        :param x:
        :param adv_x:
        :param index_to_perturb:
        :return:
        """
        x = torch.squeeze(x)
        adv_x = torch.squeeze(adv_x)
        for i in range(len(x)):
            if i not in index_to_perturb:
                if adv_x[i] != x[i]:
                    adv_x[i] = x[i]
        adv_x = torch.unsqueeze(adv_x,0)
        return adv_x



    def perturbation(self,
                     inputs:list,
                     index_to_perturb:list=None,
                     num_evasion_by_benign_content:int=0,
                     num_index_change_in_total_group:list=[],
                     num_index_change_by_gradient_group:list=[]):
        """
        inputs: the initiated inputs (with 0 initiated in perturb_index) after DOS_Extension/Content_Shift attacks
        the length of inputs should larger than original input sample

        num_evasion_w_benign_content: number of evasion times just with benign content or random bytes
        num_index_change_in_total_group: list append all num_index_change_in_total for each input malware
        num_index_change_by_gradient_group: list append with all num_byte_change_by_gradient for each input malware
        """
        ## assign label to input malware
        label = torch.tensor([1], dtype=torch.long,device=self.device)
        ## change malware 1 to benign 0, target label 0
        targeted_label = torch.tensor([0], dtype=torch.long).to(self.device)
        ## copy original input and get corresponding embed_x
        x = torch.tensor(inputs,device=self.device)
        x = torch.unsqueeze(x,0)
        embed_1 = self.model_1.embed
        embed_x_1 = embed_1(x.long()).detach()
        embed_x_1.requires_grad = True
        embed_2 = self.model_2.embed
        embed_x_2 = embed_2(x.long()).detach()
        embed_x_2.requires_grad = True

        ## get index_to_perturb, if exist, assign it to the global variable,
        ## otherwise no attack (i.e. this situation happenes when slack attack cannot find slack space)
        if len(index_to_perturb) > 0:
            self.index_to_perturb = index_to_perturb
        elif len(index_to_perturb) == 0:
            ## 1) no perturbatoin space found  --> return results based on original inputs
            ## 2) happens when the found indexes (e.g. [143432343,145645647]) are larger than the input size (102400),
            ## therefore, even though index_to_perturb exist, but it's still empty consider the input size
            self.index_to_perturb = index_to_perturb
            print('no available index found for perturb given the current input size, predicts based on original inputs')

            if self.model_2:
                (adv_x_preds_1, adv_y_1), (adv_x_preds_2, adv_y_2) = \
                    self._get_print_return_results(inputs, self.model_1, self.model_2, test_x_init=True)
                pert_size = len(self.index_to_perturb)
                return (adv_x_preds_1, adv_y_1), (adv_x_preds_2, adv_y_2), pert_size, num_evasion_by_benign_content, \
                       num_index_change_in_total_group, num_index_change_by_gradient_group
            else:
                adv_x_preds_1, adv_y_1 = self._get_print_return_results(inputs, self.model_1)
                pert_size = len(self.index_to_perturb)
                return adv_x_preds_1, adv_y_1, pert_size, num_evasion_by_benign_content, \
                       num_index_change_in_total_group, num_index_change_by_gradient_group

        # initiate input with benign content or random perturbations
        if self.pert_init_with_benign:
            benign_contents = self.get_section_content_from_folder(num_sections=100, target_section='.text')
        else:
            benign_contents = None

        ## get initiated adv_x with benign content or random bytes;
        ## and get perturbations (either benign contents or random values)
        x_adv_init,pert_init = self.apply_manipulations(inputs,
                                                        self.index_to_perturb,
                                                        self.pert_init_with_benign,
                                                        benign_contents=benign_contents)

        ## ------------------- print results for initiated x before attack and ternimate if needed -----------------------------
        ## test the initiated x that either with random value or benign content in the index_to_perturb
        (adv_x_preds_1,adv_y_1),(adv_x_preds_2,adv_y_2) = self._get_print_return_results(x_adv_init,
                                                                                         self.model_1,
                                                                                         self.model_2,
                                                                                         test_x_init=True)

        ## terminate when label already changed to benign (label=0)
        if adv_y_1 == 0 and adv_y_2 == 0:
            num_evasion_by_benign_content += 1
            return (adv_x_preds_1,adv_y_1), (adv_x_preds_2,adv_y_2), len(pert_init), num_evasion_by_benign_content, \
                   num_index_change_in_total_group, num_index_change_by_gradient_group

        ## else --> start CW based adversarial attacks
        ## update values in index_to_perturb with optimized values by FGSM
        print('\n','-'*50)
        print(f'starting CW based adversarial evasion attack...')

        ## add dim at axis=0
        x_adv = torch.tensor(np.array(x_adv_init)[np.newaxis,:],device=self.device)
        x_adv = Variable(x_adv.float(), requires_grad=True).to(self.device)

        ## get w
        ## atanh: input mush be in (-1,1), normalized first
        normalizer = util.data_normalize_inverse(x_adv,min_box=0,max_box=1)
        x_adv_norm = normalizer.data_normalize()
        x_adv_norm = torch.tensor(x_adv_norm,requires_grad=True,device=self.device)

        w = self.inverse_tanh_space(x_adv_norm).detach()
        w.requires_grad = True

        best_adv_x_1 = x_adv.clone().detach()
        best_adv_x_2 = x_adv.clone().detach()

        best_L2_1 = 1e10 * torch.ones((len(x_adv))).to(self.device)
        best_L2_2 = 1e10 * torch.ones((len(x_adv))).to(self.device)
        prev_cost = 1e10
        dim = len(x_adv.shape)
        MSELoss = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam([w], lr=self.lr)

        for step in range(self.iter_steps):
            print(f'cw step: {step+1} ...')
            # Get adversarial samples
            adv_x_norm = self.tanh_space(w)   ## w optimized each step by Adam optimizer
            adv_x = normalizer.inverse_normalize(adv_x_norm.cpu().detach().numpy(),output_shape=adv_x_norm.cpu().shape)
            adv_x = torch.tensor(adv_x,requires_grad=True,device=self.device)


            # Calculate loss
            current_L2 = MSELoss(adv_x,x.to(torch.float32)).sum(dim=1)
            L2_loss= current_L2.sum()

            embed_adv_x_1 = embed_1(adv_x.long()).detach()
            embed_adv_x_2 = embed_2(adv_x.long()).detach()
            outputs_1 = self.model_1(embed_adv_x_1)
            outputs_2 = self.model_2(embed_adv_x_2)

            f_loss_1 = self.f(outputs_1,targeted_label).sum()
            f_loss_2 = self.f(outputs_2,targeted_label).sum()

            cost_1 = L2_loss + self.c * f_loss_1
            cost_2 = L2_loss + self.c * f_loss_2

            cost = self.w_1 * cost_1 + self.w_2 * cost_2
            cost = cost.to(torch.float32)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            ## update adv_x
            _, pre_1 = torch.max(outputs_1.detach(),1)
            _, pre_2 = torch.max(outputs_2.detach(),1)
            correct_1 = (pre_1 == label).float()
            correct_2 = (pre_2 == label).float()

            mask_1 = (1-correct_1) * (best_L2_1 > current_L2.detach())
            mask_2 = (1-correct_2) * (best_L2_2 > current_L2.detach())
            best_L2_1 = mask_1 * current_L2.detach() + (1-mask_1) * best_L2_1
            best_L2_2 = mask_2 * current_L2.detach() + (1-mask_2) * best_L2_2

            mask_1 = mask_1.view([-1]+[1]*(dim-1))
            mask_2 = mask_2.view([-1]+[1]*(dim-1))

            ## should compare mask with the index_to_perturb here to
            ## avoid the useful bytes being altered
            ## need to know the example results of mask, then decide how to perfrom avoiding
            ## for all bytes not in index_to_perturb, should be all 1s
            best_adv_x_1 = mask_1*adv_x.detach()+(1-mask_1)*best_adv_x_1
            best_adv_x_2 = mask_2*adv_x.detach()+(1-mask_2)*best_adv_x_2

            ## validate best adv_x
            best_adv_x_1 = self.validate_advX(x=x,adv_x=best_adv_x_1,index_to_perturb=index_to_perturb)
            best_adv_x_2 = self.validate_advX(x=x,adv_x=best_adv_x_2,index_to_perturb=index_to_perturb)
            best_adv_x_1 = torch.squeeze(best_adv_x_1)
            best_adv_x_2 = torch.squeeze(best_adv_x_2)

            ## early stop when loss stops converge
            if step % (self.iter_steps // 2) == 0:
                if cost.item() > prev_cost:
                    ## get final adv_x based on two adv_x from two models
                    adv_x, num_index_change_in_total, num_index_change_by_gradient = \
                        self._final_adv_x_selection(x=x_adv_init,
                                                    adv_x1=best_adv_x_1,
                                                    adv_x2=best_adv_x_2,
                                                    verbose=True,
                                                    index_to_perturb=self.index_to_perturb)
                    return adv_x, num_index_change_in_total, num_index_change_by_gradient
                prev_cost = cost.item()

        ## get final adv_x based on two adv_x from two models
        adv_x, num_index_change_in_total, num_index_change_by_gradient = \
            self._final_adv_x_selection(x=x_adv_init,
                                        adv_x1=best_adv_x_1,
                                        adv_x2=best_adv_x_2,
                                        verbose=True,
                                        index_to_perturb=self.index_to_perturb)

        ## append num_index_change to list for records
        num_index_change_in_total_group.append(num_index_change_in_total)
        num_index_change_by_gradient_group.append(num_index_change_by_gradient)

        ## ------------------- print and get results for adv_x after FGSM attack -----------------------------
        ## get prediction for produced adversarial example adv_x after FGSM,
        ## print and return results (prediction score, prediction label, size of perturbation)
        (adv_x_preds_1, adv_y_1), (adv_x_preds_2, adv_y_2) = \
            self._get_print_return_results(adv_x, self.model_1, self.model_2, test_x_init=False)
        pert_size = len(self.index_to_perturb)
        return (adv_x_preds_1, adv_y_1), (adv_x_preds_2, adv_y_2), pert_size, num_evasion_by_benign_content, \
               num_index_change_in_total_group, num_index_change_by_gradient_group












