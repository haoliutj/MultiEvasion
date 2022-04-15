"""
attack base scaled up to three malware detectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os, magic, lief, random, copy

from src.util import data_normalize_inverse, reconstruction


class Attack(object):
    """
    Base class for all attacks.

    All attacks here are targeted attacks, i.e., lead model to misclassify malware samples [1] to benign samples [0],
    therefore, loss = loss_func(model(x),y_target), to minimize loss function to make model(x) as close as targeted label 0;
    - targeted attack --> set targeted = 1
    - untargeted attack --> set targeted = -1, where loss = targeted * loss(model(x),y), y is the corrected label of x
    Note: pay attention to the label (e.g., original label for untargeted attack, target label for targeted attack) when calculate loss value,
    """

    def __init__(self, model_1=None, model_2=None, model_3=None):
        """
        Initializes internal attack state.

        Arguments:
            # name (str): name of an attack.
            model (torch.nn.Module): model to attack.
            model_type: the model type among two models other than one based on raw byte  ['raw_byte', 'image']

        """

        # self.attack = name

        self.model_1 = model_1
        self.embed_1 = model_1.embed
        self.model_name_1 = str(model_1).split("(")[0]
        self.model_2 = model_2
        self.embed_2 = model_2.embed
        self.model_name_2 = str(model_2).split("(")[0]
        if model_3 != None:
            self.model_3 = model_3
            self.embed_3 = model_3.embed
            self.model_name_3 = str(model_3).split("(")[0]


        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.CrossEntropyLoss()
        self.x_box_min = 0
        self.x_box_max = 255
        self.targeted = 1   # 1 -->targeted attack: changing malware label to benign label; -1--> untargeted attack
        self.targeted_label = torch.tensor([0], dtype=torch.long).to(self.device)  # convert to benign label
        self.root_module_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        ## embed layer dictionary/matrix (used to recover embed results to before-embed byte)
        self.embed_matrix_1 = self.embed_1(torch.arange(0, 257).to(self.device))
        self.embed_matrix_2 = self.embed_2(torch.arange(0, 257).to(self.device))
        if model_3 !=None:
            self.embed_matrix_3 = self.embed_3(torch.arange(0, 256).to(self.device))

        ## initiate an empty log file for recording results
        output_path = '../result/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.log_file_for_result = output_path + 'log_file_for_result.txt'
        if os.path.exists(self.log_file_for_result):
            os.remove(self.log_file_for_result)


    def get_section_content_from_folder(self,
                                        folder_path=None,
                                        num_sections=50,
                                        target_section=".text",
                                        min_extract_size=512) -> (list):
        """
        extract content from sections of benign PE files

        folder_path: path of goodwares. Default as
        num_sections: the number of benign content to extract
        target_section: the section to extract content from. default as ".text"
        min_extract_size: extract the benign content that the size larger than min_extract_size, otherwise discard

        return: group of list of section content;
        """
        extract_sections = []       # a list of list of section content
        # sectionName_fileName = []   # a group of list of section name and the corresponding file name
        counter = 0

        if not folder_path:
            folder_path = self.root_module_path + '/data/goodware_filelist.csv'

        file_to_extract = pd.read_csv(folder_path,header=None,index_col=0).index
        file_path = self.root_module_path + '/data/all_file/'
        for filename in file_to_extract:
            path = os.path.join(file_path,filename)
            if "PE" not in magic.from_file(path):
                continue
            liefpe = lief.PE.parse(path)
            for section in liefpe.sections:
                if section.name.startswith(target_section):
                    if min_extract_size and len(section.content) < min_extract_size:
                        continue
                    extract_sections.append(section.content)
                    # sectionName_fileName.append((section.name,filename))
                    counter += 1
            if counter >= num_sections:
                break

        return extract_sections


    def apply_manipulations(self,input, index_to_perturb, pert_init_with_benign=True, benign_contents=None) -> (list):
        """
        initiate the perturbations with benign content or random initiate

        input: list of raw bytes, initiated with 0s
        pert_init_with_benign: bool, default as benign, otherwise random initiate
        index_to_perturb: list of index that can perturb
        benign_contents: group of list of benign contents

        return:
            - initiated inputs with benign content in perturbation index
            - initiated perturbations
        """
        pert_length = len(index_to_perturb)

        ## initiate input with benign
        if pert_init_with_benign:
            ## random select one benign content from the benign contents (shape=[batch=100,length of each content]),
            ## with length larger than pert_length if possible
            benign_content = random.choice(benign_contents)
            if len(benign_content) < pert_length:
                iter_count = 0
                ## continue randomly select benign content, to select one with length >perturbation length
                ## at most iterate 50 times.
                ## if can not get larger length than pertubation length after 50 times, append with 0s
                while len(benign_content) < pert_length and iter_count < 50:
                    benign_content = random.choice(benign_contents)
                    iter_count += 1
                if len(benign_content) < pert_length:
                    pert_init = benign_content + [0 for _ in range(pert_length-len(benign_content))]
                else:
                    pert_init = benign_content[:pert_length]
            else:
                pert_init = benign_content[:pert_length]

        ## initiate input with random bytes
        else:
            pert_init = np.random.randint(0,255,pert_length)

        ## replace perturbation index with perturbations (benign content or random values)
        X_init_w_pert = copy.deepcopy(input)
        for i,p in enumerate(index_to_perturb):
            X_init_w_pert[p] = pert_init[i]

        return X_init_w_pert, pert_init


    def _forward(self, input, model):
        """
        forward process of NN model to get predictions
        input: [N,W] <==> [batch,width] should be long format since it feed into embed layer first
        model: target model

        raw_byte_input: True, the model is based on raw byte, which means it has a embed layer;
        otherwise, the model is based on image, no embed layer

        return:
            - ndarray: [a,b], predictions
            - embeded x after emebed layer. float, require_grad=True

        note: "_" in the name of function indicates that it won't be import by "import *"
        """
        embed = model.embed
        embed_x = embed(input.long()).detach()
        embed_x.requires_grad = True    # Ture --> 加入到反向传播中计算梯度
        output = model(embed_x.to(self.device))
        if not output.shape == (1,2):
            output=torch.unsqueeze(output,dim=0)
        preds = F.softmax(output,dim=1)
        # preds = preds.detach().numpy().squeeze()

        return preds, embed_x


    def _get_preds_label(self,input:list,model_1=None,model_2=None,model_3=None):
        """
        get the label for the input toward models

        input: list or torch.tensor
        model_1: pytorch model
        model_2: pytorch model
        Note that, when two models based on different input format, model_1 default based on
        raw bytes, model_2 default based on image

        return: prediction scores  and correponding label
            - preds_score,label; or
            - (preds_score_1,y_1),(preds_score_2,y_2)
        """
        if isinstance(input,np.ndarray):
            X = Variable(torch.tensor(input[np.newaxis, :], dtype=float, device=self.device))
        else:
            X = input
        preds_score_1, _ = self._forward(X, model_1)
        y_1 = torch.argmax(preds_score_1, 1).item()
        if model_2:
            preds_score_2, _ = self._forward(X, model_2)
            y_2 = torch.argmax(preds_score_2, 1).item()
            if model_3:
                preds_score_3, _ = self._forward(X, model_3)
                y_3 = torch.argmax(preds_score_3, 1).item()
                return (preds_score_1, y_1), (preds_score_2, y_2),(preds_score_3,y_3)
            else:
                return (preds_score_1,y_1),(preds_score_2,y_2)
        else:
            return preds_score_1,y_1


    def _get_print_return_results(self,input:list,model_1=None,model_2=None,model_3=None,test_x_init=False):
        """
        get the prediction scores toward model/models,
        print results,
        return results

        input: list
        model_1: pytorch model
        model_2: pytorch model
        test_x_init: True--> indicates working on the initiated input. i.e., initiated x that either filled with
        random value or benign content. Default as False. (Difference: print results indicated as initiated adv_x)

        return: prediction scores  and correponding label
            - preds_score,label; or
            - (preds_score_1,y_1),(preds_score_2,y_2)
        """
        if model_3:
            (x_preds_1, y_1), (x_preds_2, y_2),(x_preds_3,y_3) = self._get_preds_label(input,
                                                                                       model_1=model_1,
                                                                                       model_2=model_2,
                                                                                       model_3=model_3)
            if test_x_init:
                print(f"Initiated Adv_x, Model: {self.model_name_1}, Score: {x_preds_1.cpu().squeeze().detach().numpy()}, Label: {y_1}")
                print(f"Initiated Adv_x, Model: {self.model_name_2}, Score: {x_preds_2.cpu().squeeze().detach().numpy()}, Label: {y_2}")
                print(f"Initiated Adv_x, Model: {self.model_name_3}, Score: {x_preds_3.cpu().squeeze().detach().numpy()}, Label: {y_3}")

                if y_1 == 0 and y_2 == 0 and y_3==0:
                    print(f"------Evading Three models successfully with benign content initiation! Stop Evading!------")

                return (x_preds_1, y_1), (x_preds_2, y_2),(x_preds_3,y_3)

            else:
                print(f"Attack {self.attack}, Model: {self.model_name_1}, Score: {x_preds_1.squeeze().cpu().detach().numpy()}, Label: {y_1}")
                print(f"Attack {self.attack}, Model: {self.model_name_2}, Score: {x_preds_2.squeeze().cpu().detach().numpy()}, Label: {y_2}")
                print(f"Attack {self.attack}, Model: {self.model_name_3}, Score: {x_preds_3.squeeze().cpu().detach().numpy()}, Label: {y_3}")
                if y_1 == 0 and y_2 == 0 and y_3 == 0:
                    print(f"------Evading Three models successfully! Stop Evading!------")

                return (x_preds_1, y_1), (x_preds_2, y_2),(x_preds_3,y_3)
        elif model_2:
            (x_preds_1,y_1),(x_preds_2,y_2) = self._get_preds_label(input,model_1=model_1,model_2=model_2)

            if test_x_init:
                print(f"Initiated Adv_x, Model: {self.model_name_1}, Score: {x_preds_1.cpu().squeeze().detach().numpy()}, Label: {y_1}")
                print(f"Initiated Adv_x, Model: {self.model_name_2}, Score: {x_preds_2.cpu().squeeze().detach().numpy()}, Label: {y_2}")

                if y_1 == 0 and y_2 == 0:
                    print(f"------Evading multiple models successfully! Stop Evading!------")

                return (x_preds_1, y_1), (x_preds_2, y_2)

            else:
                print(f"Attack {self.attack}, Model: {self.model_name_1}, Score: {x_preds_1.squeeze().cpu().detach().numpy()}, Label: {y_1}")
                print(f"Attack {self.attack}, Model: {self.model_name_2}, Score: {x_preds_2.squeeze().cpu().detach().numpy()}, Label: {y_2}")
                if y_1 == 0 and y_2 == 0:
                    print(f"------Evading multiple models successfully! Stop Evading!------")

                return (x_preds_1, y_1), (x_preds_2, y_2)
        else:
            x_preds_1, y_1 = self._get_preds_label(input,model_1=model_1)

            if test_x_init:
                print(f"Initiated Adv_x, Model: {self.model_name_1}, Score: {x_preds_1.squeeze().cpu().detach().numpy()}, Label: {y_1}")

                if y_1 == 0:
                    print(f"------Evading model (single) successfully! Stop Evading!------")
                return x_preds_1, y_1
            else:
                print(f"Attack {self.attack}, Model: {self.model_name_1}, Score: {x_preds_1.squeeze().cpu().detach().numpy()}, Label: {y_1}")
                if y_1 == 0:
                    print(f"------Evading model (single) successfully! Stop Evading!------")
                return x_preds_1,y_1


    def _get_grad_sign(self, input, index_to_perturb:list):
        """
        get sign of grad (e.g. in FGSM)
        input: tensor, get the gradient of input (e.g. embed_x). (i.e., backward w.r.t the input to calculate gradient)
        index_to_perturb: the index that can be modified or need to get the sign

        return: sign of the gradient in the perturbed index
        """
        grad = input.grad
        grad_sign = grad.detach().sign()[0]
        grad_sign_for_perturb_index = [grad_sign[i].cpu().numpy() for i in index_to_perturb]
        grad_sign_for_perturb_index = np.array(grad_sign_for_perturb_index)

        return grad_sign_for_perturb_index


    def _get_final_grad_sign(self,grad_sign_1=None,grad_sign_2=None,grad_sign_3=None,overshot=0.1):
        """
        obtain final sign if two signs from two models are different

        add a overshot when each element has the same sign:
        if two signs are same in positive direction, then add +overshot
        if two signs are same in negative direction, then add -overshot
        if two signs are both 0, then keep as 0

        grad_sign: a list of sign, e.g. [1,-1,1,1,-1,0,1,0], (only include 3 unique elements, -1,-0,+1)
        overshot: for boosting minimization/maxmization of loss value

        return: the sum array; the effective length of perturbation (exclude neural (i.e. sign=0) index)
        """
        out = []

        for i in range(len(grad_sign_1)):
            temp = grad_sign_1[i] + grad_sign_2[i]

            ## same sign in positive diretion, add +overshot
            if temp == 2:
                temp = 1 + overshot
            ## same sign in negative diretion, add -overshot
            elif temp == -2:
                temp = -1 - overshot
            ## no sign/neutal --> keep neural 0
            else:
                temp = 0
            out.append(temp)

        # the effective perturbation length/bytes (exclude neural/no sign index)
        len_perturb_bytes = sum(abs(np.array(out)))
        print(f"perturbation length is {len_perturb_bytes}")

        return np.array(out), len_perturb_bytes



    def _get_advX(self,x=None,embed_x=None,sign=None,eps=None,embed_matrix=None,index_to_perturb=None):
        """
        obtain adversarial malware by add pert on input: operate based on the embedded shape level ([1,102400,8]),
        and get pre-embed result ([1,102400])

        sign (from embed layer, shape[1,102400,8]) --> pert=eps*sign--> normalized adv_x (shape=[1,102400,8]) = normalized(embed x) + (-pert)
        --> reverse normalized adv_x --> reverse embedded normalized adv_x (-->shape[102400] 1-d)

        Note: perturbation is based on gradient descent direction, which is the direction to minimize the loss
              goal here is to maximize loss to make the prediction to other label (instead original label)
              should add gradient in its increase direction (i.e., -gradient_sign*eps)

        :param x: original input binary file (shape=[batch,first_n_bytes])
        :param embed_x: embedded results of original input binary file (wait for crafting) (shape=[batch,first_n_bytes,embed_size])
        :param sign: grad sign of model (shape=[batch,first_n_bytes,embed_size])
        :param eps:-->float: weight for perturbation
        :param embed_layer: embed layer network to get the embedded results
        :param embed_matrix: embed matrix/dictionary to map embedded results back to pre-embed
        :return: -->ndarray: adversarial/crafted x (shape=[first_n_bytes] 1-d)
        """
        ## normalized embed_x to [0,1]. (shape[batch=1,first_n_bytes,embed_size])
        normalizer = data_normalize_inverse(embed_x)
        embed_x_norm = normalizer.data_normalize()
        embed_x_norm = np.squeeze(embed_x_norm)

        ## get pert by multiple eps. (shape[batch=1,first_n_bytes,embed_size])
        pert = eps * sign   # (0,1)

        ## get normalized embed_adv_x_norm_in_perturb_index by adding pert on input based on pertubation index.
        ## i.e. final value after applied perturbations in the position of perturb index
        ## --> (shape[batch=1,length of index_to_perturb,embed_size])
        embed_adv_x_norm_in_pertrub_index = np.zeros(pert.shape)
        for i,p in enumerate(index_to_perturb):
            ## perturbation is based on gradient desecent, which is the direction to minimize the loss
            ## goal here is to maxmize loss to make the prediction to other label
            ## should add gradient increase direction (-gradient_sign)
            embed_adv_x_norm_in_pertrub_index[i,:] = embed_x_norm[p] + (-pert[i])

        ## reverse normalized embed_adv_x_norm_in_perturb_index to pre-norm.
        ## --> shape[batch=1,first_n_bytes,embed_size]
        embed_adv_x_in_perturb_index = normalizer.inverse_normalize(embed_adv_x_norm_in_pertrub_index,
                                                                    output_shape=embed_adv_x_norm_in_pertrub_index.shape)

        ## reverse embeding process back to pre-embed. --> shape[batch=1,length of index_to_perturb]
        ## reconstrcution compution on cpu faster than on gpu
        adv_x_in_perturb_index = reconstruction(torch.from_numpy(embed_adv_x_in_perturb_index).to('cpu'),
                                                embed_matrix.to('cpu')).detach().numpy()

        ## get adv_x by updating final value in perturb index to orginal input x
        adv_x = self._update_input(input=x,index_to_perturb=index_to_perturb,pert=adv_x_in_perturb_index)

        ## clamp to [0, 256]
        adv_x = np.clip(adv_x, self.x_box_min, self.x_box_max)

        return adv_x


    def _get_advX_for_FFGSM(self,x=None,embed_x=None,sign=None,eps=None,embed_matrix=None,index_to_perturb=None,alpha=10/255):
        """
        obtain adversarial malware by add pert on input: operate based on the embedded shape level ([1,102400,8]),
        and get pre-embed result ([1,102400])

        sign (from embed layer, shape[1,102400,8]) --> pert=eps*sign--> normalized adv_x (shape=[1,102400,8]) = normalized(embed x) + (-pert)
        --> reverse normalized adv_x --> reverse embedded normalized adv_x (-->shape[102400] 1-d)

        Note: perturbation is based on gradient descent direction, which is the direction to minimize the loss
              goal here is to maximize loss to make the prediction to other label (instead original label)
              should add gradient in its increase direction (i.e., -gradient_sign*eps)

        :param x: original input binary file (shape=[batch,first_n_bytes])
        :param embed_x: embedded results of original input binary file (wait for crafting) (shape=[batch,first_n_bytes,embed_size])
        :param sign: grad sign of model (shape=[batch,first_n_bytes,embed_size])
        :param eps:-->float: weight for perturbation
        :param embed_layer: embed layer network to get the embedded results
        :param embed_matrix: embed matrix/dictionary to map embedded results back to pre-embed
        :return: -->ndarray: adversarial/crafted x (shape=[first_n_bytes] 1-d)
        """
        x = copy.deepcopy(x)
        ## normalized embed_x to [0,1]. (shape[batch=1,first_n_bytes,embed_size])
        normalizer = data_normalize_inverse(embed_x)
        embed_x_norm = normalizer.data_normalize()
        embed_x_norm = np.squeeze(embed_x_norm)

        ## get pert by multiple eps. (shape[batch=1,first_n_bytes,embed_size])
        pert = eps * sign   # (0,1)

        ## get normalized embed_adv_x_norm_in_perturb_index by adding pert on input based on pertubation index.
        ## i.e. final value after applied perturbations in the position of perturb index
        ## --> (shape[batch=1,length of index_to_perturb,embed_size])
        embed_adv_x_norm_in_pertrub_index = np.zeros(pert.shape)
        for i,p in enumerate(index_to_perturb):
            ## perturbation is based on gradient desecent, which is the direction to minimize the loss
            ## goal here is to maxmize loss to make the prediction to other label
            ## should add gradient increase direction (-gradient_sign)
            embed_adv_x_norm_in_pertrub_index[i,:] = embed_x_norm[p] + (-pert[i])

        ## reverse normalized embed_adv_x_norm_in_perturb_index to pre-norm.
        ## --> shape[batch=1,first_n_bytes,embed_size]
        embed_adv_x_in_perturb_index = normalizer.inverse_normalize(embed_adv_x_norm_in_pertrub_index,
                                                                    output_shape=embed_adv_x_norm_in_pertrub_index.shape)

        ## reverse embeding process back to pre-embed. --> shape[batch=1,length of index_to_perturb]
        ## reconstrcution compution on cpu faster than on gpu
        adv_x_in_perturb_index = reconstruction(torch.from_numpy(embed_adv_x_in_perturb_index).to('cpu'),
                                                embed_matrix.to('cpu')).detach().numpy()

        ## get adv_x by updating final value in perturb index to orginal input x
        adv_x = self._update_input(input=x,index_to_perturb=index_to_perturb,pert=adv_x_in_perturb_index)
        delta = np.clip((adv_x - x)/255, -alpha, alpha)
        adv_x = np.clip(x/255 + delta, 0, 1) * 255

        return adv_x


    def _get_advX_for_PGD(self,
                          x=None,
                          embed_x_init=None,
                          embed_x_adv=None,
                          sign=None,
                          eps=None,
                          alpha=None,
                          embed_matrix=None,
                          index_to_perturb=None):
        """
        obtain adversarial malware by add pert on input: operate based on the embedded shape level ([1,102400,8]),
        and get pre-embed result ([1,102400])

        sign (from embed layer, shape[1,102400,8]) --> pert=alpha*sign
        --> normalized adv_x (shape=[1,102400,8]) = normalized(embed x) + (-pert)
        --> clip accumulate normalized perturbation inside [-eps,eps], clip(accumulate_pert_norm,-eps,eps),
        where:accumulate_pert_norm = adv_x - embed_x_adv_norm, embed_x_adv_norm: original input after embedded w/o any perturbation
        --> reverse normalized adv_x --> reverse embedded normalized adv_x (-->shape[102400] 1-d)

        Note: perturbation is based on gradient descent direction, which is the direction to minimize the loss
              goal here is to maximize loss to make the prediction to other label (instead original label)
              should add gradient in its increase direction (i.e., -gradient_sign*eps)

        :param x: original input binary file (with benign content or random values initiation) (shape=[batch,first_n_bytes])
        :param embed_x_adv: embedded results of adv_x during each iteration (shape=[batch,first_n_bytes,embed_size])
        :param embed_x_init: embedded results of original input x(i.e. initiated x with benign contents )
        :param sign: grad sign of model (shape=[batch,first_n_bytes,embed_size])
        :param alpha:-->float: maximum perturbation. (default:0.3)
        :param eps:-->float: perturbation step size (default:0.07)
        :param embed_layer: embed layer network to get the embedded results
        :param embed_matrix: embed matrix/dictionary to map embedded results back to pre-embed
        :return: -->ndarray: adversarial/crafted x (shape=[first_n_bytes] 1-d)
        """
        ## normalize embed_x_init to [0,1]
        normalizer_x_init = data_normalize_inverse(embed_x_init)
        embed_x_init_norm = normalizer_x_init.data_normalize()
        embed_x_init_norm = np.squeeze(embed_x_init_norm)

        ## normalized embed_x_adv to [0,1]. (shape[batch=1,first_n_bytes,embed_size])
        normalizer = data_normalize_inverse(embed_x_adv)
        embed_x_adv_norm = normalizer.data_normalize()
        embed_x_adv_norm = np.squeeze(embed_x_adv_norm)

        ## get pert by multiple eps. (shape[batch=1,first_n_bytes,embed_size])
        pert = eps * sign   # (0,1)

        ## get normalized embed_adv_x_norm_in_perturb_index by adding pert on input based on pertubation index.
        ## i.e. final value after applied perturbations in the position of perturb index
        ## --> (shape[batch=1,length of index_to_perturb,embed_size])
        embed_adv_x_norm_in_pertrub_index = np.zeros(pert.shape)
        embed_x_init_norm_in_pertrub_index = np.zeros(pert.shape)
        for i,p in enumerate(index_to_perturb):
            ## perturbation is based on gradient desecent, which is the direction to minimize the loss
            ## goal here is to maxmize loss to make the prediction to other label
            ## should add gradient increase direction (-gradient_sign)
            embed_x_init_norm_in_pertrub_index[i,:] = embed_x_init_norm[p]
            embed_adv_x_norm_in_pertrub_index[i,:] = embed_x_adv_norm[p] + (-pert[i])

        ## limit accumulated perturbation to the range [-alpha,alpha] in each iteration
        accumulate_pert = np.clip(embed_adv_x_norm_in_pertrub_index - embed_x_init_norm_in_pertrub_index, -alpha, alpha)
        embed_adv_x_norm_in_pertrub_index = np.clip(embed_x_init_norm_in_pertrub_index + accumulate_pert, 0, 1)

        ## reverse normalized embed_adv_x_norm_in_perturb_index to pre-norm.
        ## --> shape[batch=1,first_n_bytes,embed_size]
        embed_adv_x_in_perturb_index = normalizer.inverse_normalize(embed_adv_x_norm_in_pertrub_index,
                                                                    output_shape=embed_adv_x_norm_in_pertrub_index.shape)

        ## reverse embeding process back to pre-embed. --> shape[batch=1,length of index_to_perturb]
        ## reconstrcution compution on cpu faster than on gpu
        adv_x_in_perturb_index = reconstruction(torch.from_numpy(embed_adv_x_in_perturb_index).to('cpu'),
                                                embed_matrix.to('cpu')).detach().numpy()

        ## get adv_x by updating final value in perturb index to orginal input x
        adv_x = self._update_input(input=x,index_to_perturb=index_to_perturb,pert=adv_x_in_perturb_index)

        ## clamp to [0, 255]
        adv_x = np.clip(adv_x, self.x_box_min, self.x_box_max)

        return adv_x


    def _final_adv_x_selection(self, x=None, adv_x1=None, adv_x2=None, adv_x3=None,verbose=True, index_to_perturb=None):
        """
        get the final adv_x by compare adv_x1 and adv_x2 with x.
        x: initiated with benign contents or random bytes in the positions created by DOS/content_shift attacks

        e.g.,   x  =        [2,77,4,32,10],
                adv_x1  =   [5,60,8,36,3],
                adv_x2  =   [9,80,16,12,7]
            --> adv_x   =   [9,77,16,32,3]

        :param x -->ndarray: original input of binary file (here is the x initiated with benign content or
        random values on DOS/Shift attacks indexes) -->shape=[first_n_bytes+extension_attack_bytes], 1-d
        :param adv_x1 -->ndarray: adversarial malware 1 -->shape=[first_n_bytes+extension_attack_bytes], 1-d
        :param adv_x2 -->ndarray: adversarial malware 2 -->shape=[first_n_bytes+extension_attack_bytes], 1-d
        :param verbal -->bool: whether output statistics (number of bytes changed by gradient optimized values)
        :param index_to_perturb -->list: index of position for perturbation
        :return: -->ndarray: final adversarial malware -->shape=[first_n_bytes+extension_attack_bytes], 1-d
        """
        adv_x = copy.deepcopy(x)
        count = 0
        if len(adv_x3) > 0:
            ## get final adv_x for three detectors
            for i in range(len(x)):
                if adv_x1[i] > x[i] and adv_x2[i] > x[i] and adv_x3[i] > x[i]:
                    adv_x[i] = max(adv_x1[i], adv_x2[i], adv_x3[i])
                    count += 1
                elif adv_x1[i] < x[i] and adv_x2[i] < x[i] and adv_x3[i] < x[i]:
                    adv_x[i] = min(adv_x1[i], adv_x2[i], adv_x3[i])
                    count += 1
                else:
                    continue
        else:
            ## get final adv_x for two detectors
            for i in range(len(x)):
                if adv_x1[i] > x[i] and adv_x2[i] > x[i]:
                    adv_x[i] = max(adv_x1[i], adv_x2[i])
                    count += 1
                elif adv_x1[i] < x[i] and adv_x2[i] < x[i]:
                    adv_x[i] = min(adv_x1[i], adv_x2[i])
                    count += 1
                else:
                    continue

        ## output number of bytes changing statisitics
        ## output the number of bytes that changed by gradients optimized value
        if verbose:

            ## print result
            print(f'{len(index_to_perturb)} of bytes crafted in total.')
            print(f'{count} of bytes changed by gradients optimized values.')
            print(f'{len(index_to_perturb) - count} of bytes crafted by benign contents (or random bytes).')
            print('\n')

            ## output result to file
            output_path = '../result/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(output_path+'bytes_changed_by_gradient.txt','w') as f:
                print(f'{len(index_to_perturb)} of bytes crafted in total.', file=f)
                print(f'{count} of bytes changed by gradients optimized values.', file=f)
                print(f'{len(index_to_perturb)-count} of bytes crafted by benign contents (or random bytes).', file=f)
                print('\n',file=f)

        ## return final adv_x, num_index_change_in_total,num_index_changed_by_gradient
        return adv_x,len(index_to_perturb),count


    def _get_final_pert(self,pert1:list, pert2:list) ->torch.Tensor:
        """
        choose the minimum value in each index (consider max 256 is padding symbol)

        pert1: the finalized perturbation based on model 1
        pert2: the finalized perturbation based on model 2

        return: a Tensor of processed perturbations
        """
        pert = [0]* len(pert1)

        for i in range(len(pert1)):
            pert[i] = min(pert1[i],pert2[i])

        return pert



    def _recover_embed_results(self,x=None,embed_matrix=None):
        """
        recover embed results back to original value (before-embed),
        e.g., embed results x=[[0,1,0,0,1,0,2,1]], shape=[1,8]
        --> before-embed value: x0=7, shape=[1]

        :param x: --> Tensor: the embed result, which need to be recovered
        :param embed_matrix: embed dictionary/matrix used to recover embed results
        :return:--> Tensor: before-embed value
        """
        ## recovering/reconstruction
        ori_x = reconstruction(x.to(self.device), embed_matrix)

        return ori_x



    def _update_pert_with_normalization_and_reconstruction(self,input:list, pert_step, embed):
        """
        normalize input to [0,1] --> update perturbation --> inverse it back

        pert_init --> embed layer --> normalization --> add pert_step --> inverse normalization
        --> inverse embed process to [0,256]

        input: the initiated perturbation
        pert_step: the perturbation values after backward function, used to add to original sample
        embed: the corresponding embed layer

        return: a tensor of updated perturbations [0,256]
        """
        pert_embed = embed(torch.from_numpy(input).long()).to(self.device)

        ## normalize embed of initiated perturbation to [0,1]
        normalizer = data_normalize_inverse(pert_embed)
        pert_norm = normalizer.data_normalize()

        ## update perturbation, must add, not minus since already consider target attack or not when calculate loss
        pert = (pert_norm + pert_step.data.cpu().numpy())

        ## inverse normalization process
        pert = normalizer.inverse_normalize(pert)

        ## inverse embed process to [0,256]
        embed_matrix = embed(torch.arange(0, 257).to(self.device))
        pert = reconstruction(torch.Tensor(pert).to(self.device), embed_matrix).detach().numpy()

        # clamp to [0,256]
        pert = torch.clamp(pert,self.x_box_min,self.x_box_max)

        return pert


    def _update_input(self, input=None, index_to_perturb=None, pert=None):
        """
        update input with finalized perturbations in the position of index_to_perturb

        input: list of raw bytes
        index_to_perturb ->list: positions for perturbation
        pert: list of finalized perturbations

        return: list of updated input with finalized perturbations
        """
        adv_x = copy.deepcopy(input)
        for i,p in enumerate(index_to_perturb):
            adv_x[p] = pert[i]

        return adv_x


    def _add_pert_in_certain_index(self, X:list=None, pert:list=None, index_to_perturb:list=None):
        """
        add perturbation to X in the index of index_to_perturb.
        X and pert are normalized. where X: [0,1], pert: [-1.0,1.0]
        :param X:
        :param pert:
        :param index_to_perturb:
        :return:
        """
        for p in index_to_perturb:
            X[p] = X[p] + pert[p]

        X = np.clip(X,0,1)

        return X


















