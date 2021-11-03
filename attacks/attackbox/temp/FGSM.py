import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from attacks.attackbox.attack_base import Attack



class FGSM(Attack):
    """
    FSGM based on TWO Models: can only take one input sample each time
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
        - inputs: `(N, W)` where `N = number of batches`,  `W = width`. N has to be 1 !!
        - labels: `(N)` where `N = number of batches`, and each value is the corresponding label
        - output: :math:`(N, W)`.

    Examples:
        > attack = attackbox.FGSM(model_1,model_2,eps=0.007,index_to_perturb=list)
        > adv_Xs = attack(inputs, labels)

    return:
        - if one target model: prediction scores, label of adversarial malware, size of perturbation (i.e., length of pert)
        - if two target models: (prediction_scores_1, adv_y_1), (prediction_scores_2, adv_y_2),size of perturbation (i.e., length of pert)

    """

    def __init__(self,
                 model_1,
                 model_2=None,
                 eps: float = 0.07,
                 w_1: float = 0.5,
                 w_2: float = 0.5,
                 index_to_perturb: list = None, # can also obtain in perturbation step
                 random_init: bool = True,
                 pert_init_with_benign: bool = True):

        super(FGSM, self).__init__("FGSM", model_1, model_2)

        if not index_to_perturb:
            index_to_perturb = [i for i in range(2, 0x3c)] ## full dos: [2,0x3c) and [0x3c,0x40]
            print("Crafting based on partial DOS header")

        self.w_1, self.w_2 = w_1, w_2
        self.eps = eps
        self.index_to_perturb = index_to_perturb
        self.random_init = random_init
        self.pert_init_with_benign = pert_init_with_benign


    def perturbation(self, inputs:list, labels=None,index_to_perturb=None):
        """
        inputs: the initiated inputs (with 0) after DOS_Extension/Content_Shift attacks
        the length of inputs should larger than original input sample
        """

        # get index_to_perturb; Default as partial DOS attack
        if not index_to_perturb:
            self.index_to_perturb = index_to_perturb

        # initiate input with benign content or random perturbations
        if self.pert_init_with_benign:
            benign_contents = self.get_section_content_from_folder(num_sections=100, target_section='.text')
        else:
            benign_contents = None
        x_adv_init,pert_init = self.apply_manipulations(inputs, self.index_to_perturb,
                                                         self.pert_init_with_benign, benign_contents=benign_contents)

        # ------------------- print results for initiated x before attack and ternimate if needed -----------------------------
        # test the initiated x that either with random value or benign content in the index_to_perturb
        # and terminate when label already changed to benign
        _, _ = self._get_print_return_results(x_adv_init, self.model_1, self.model_2, test_x_init=True)

        x_adv = torch.tensor(np.array(x_adv_init)[np.newaxis,:],device=self.device)

        # ------------- apply gaussian noise or others on input, which can improve the performance ----------------
        # https://www.cnblogs.com/tangweijqxx/p/10615950.html
        if self.random_init:
            x_adv = x_adv + torch.Tensor(np.random.uniform(-self.eps, self.eps, x_adv.shape)).\
                type_as(x_adv).to(self.device)
        x_adv = Variable(x_adv.float(), requires_grad=True).to(self.device)

        "----------- Here: may need to cut the size of input to fixed number since the DOS/Shift attack increase length-----"

        # forward process
        preds_1, embed_x_1 = self._forward(x_adv, self.model_1) # embed_x: float
        loss_1 = self.targeted* self.loss(preds_1, self.targeted_label)

        # against 1 or 2 models
        if not self.model_2:
            loss = loss_1
        else:
            preds_2, embed_x_2 = self._forward(x_adv, self.model_2)
            loss_2 = self.targeted * self.loss(preds_2, self.targeted_label)
            loss = self.w_1 * loss_1 + self.w_2 * loss_2

        # zero_grad before backward, otherwise it will accumulate gradients in mini-batches
        self.model_1.zero_grad()
        if self.model_2:
            self.model_2.zero_grad()
        loss.backward()

        # get sign and perturbation
        grad_sign_for_perturb_index_1 = self._get_grad_sign(embed_x_1,self.index_to_perturb)

        ## reconstruct/recover embed results back to pre-embed value
        pre_embed_of_grad_sign_for_perturb_index_1 = self._recover_embed_results(torch.from_numpy(grad_sign_for_perturb_index_1),self.embed_matrix_1)

        ## obtain final pertbation based on the either one model or two models
        ## one model: simple output pert = eps * sign
        ## two models: 1) if sign1=sign2 --> ... 2) else --> ...
        if self.model_2:
            grad_sign_for_perturb_index_2 = self._get_grad_sign(embed_x_2,self.index_to_perturb)

            if grad_sign_for_perturb_index_1 == grad_sign_for_perturb_index_2:
                print(f'sign on two models is the same')
                grad_sign_for_perturb_index = grad_sign_for_perturb_index_1
            else:
                print(f"sign of the gradient on two models is different")
                grad_sign_for_perturb_index = self._get_final_grad_sign(grad_sign_for_perturb_index_1,grad_sign_for_perturb_index_2,overshot=0.2)
        else:
            grad_sign_for_perturb_index = grad_sign_for_perturb_index_1

        pert_step = self.eps * grad_sign_for_perturb_index

        #-----------may need to revise for how to choose the final pert between two ----------
        # update perturbation after embed layer
        """
        pert_init --> embed layer --> normalization --> add pert --> inverse normalization
        --> inverse embed process to [0,256] 
        """
        pert_updated_1 = self._update_pert_with_normalization_and_reconstruction(pert_init,pert_step,self.embed_1)
        if self.model_2:
            pert_updated_2 = self._update_pert_with_normalization_and_reconstruction(pert_init,pert_step,self.embed_2)
            if list(pert_updated_1) == list(pert_updated_2):
                pert_updated = pert_updated_1
            else:
                pert_updated = self._get_final_pert(list(pert_updated_1),list(pert_updated_2))
        else:
            pert_updated = pert_updated_1
        pert_size = len(pert_updated)   # total pert_size bytes added
        # -----------may need to revise for how to choose the final pert between two ----------

        # get final adv_x: update input with finalized perturbations
        x_adv = self._update_input(x_adv_init,self.index_to_perturb,pert_updated)

        # ------------------- print and get results for adv_x after FGSM attack -----------------------------
        # get prediction for produced adversarial example x_adv after FGSM and print and return results
        if self.model_2:
            (adv_x_preds_1,adv_y_1),(adv_x_preds_2,adv_y_2) = self._get_print_return_results(x_adv,self.model_1,self.model_2,test_x_init=False)
            return (adv_x_preds_1,adv_y_1),(adv_x_preds_2,adv_y_2),pert_size
        else:
            adv_x_preds_1, adv_y_1 = self._get_print_return_results(x_adv[np.newaxis,:],self.model_1)
            return adv_x_preds_1, adv_y_1,pert_size







