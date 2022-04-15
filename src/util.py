import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from PIL import Image
from torchvision import transforms
import os



def write_pred(test_pred,test_idx,file_path):
    test_pred = [item for sublist in test_pred for item in sublist]
    with open(file_path,'w') as f:
        for idx,pred in zip(test_idx,test_pred):
            print(idx.upper()+','+str(pred[0]),file=f)



class ExeDataset(Dataset):
    """
    load a batch of data each time instead loading all, to reduce the workload on the memory

    padding with 0s eventhough 0 already used in the malware files.
    the reason use 0 is because 0 won't render gradient, plus if use 256,
    which is out of the range ([0,255]) for bytearray function
    """
    def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000,padding_symbol=0):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte
        self.padding_symbol = padding_symbol

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        """
        use padding_symbol value as padding symbol;
        """
        try:
            with open(self.data_path+self.fp_list[idx],'rb') as f:
                ## padding with 0, in order to keep value in the range [0,256)
                ## the reason use 0 is because 0 won't render graident, plus 256 out of the range for bytearray
                if self.padding_symbol == 0:
                    tmp = [i for i in f.read()[:self.first_n_byte]]   # convert to [0,255], normally [0,255] 0: grey, 255: white
                # padding with assigned value
                else:
                    tmp = [i for i in f.read()[:self.first_n_byte]]  # convert to [0,255],0: grey, 255: white; 256 as padding
                tmp = tmp+[self.padding_symbol]*(self.first_n_byte-len(tmp))
        except:
            ## read file with Captize Name
            with open(self.data_path + self.fp_list[idx].lower(), 'rb') as f:
                # padding with 0
                if self.padding_symbol == 0:
                    tmp = [i for i in f.read()[:self.first_n_byte]]  # convert to [0,255], normally [0,255] 0: grey, 255: white
                # padding with assigned value
                else:
                    tmp = [i for i in f.read()[:self.first_n_byte]]  # convert to [0,255],0: grey, 255: white; 256 as padding
                tmp = tmp + [self.padding_symbol] * (self.first_n_byte - len(tmp))
        x = np.array(tmp)
        y = np.array([self.label_list[idx]])

        return x,y









class ExeDataset_Malware(Dataset):
    """
    only load malware samples
    """
    def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000,padding_symbol=256):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte
        self.padding_symbol = padding_symbol

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        if self.fp_list[idx].startswith('mal'):
            print(self.fp_list[idx])
            with open(self.data_path+self.fp_list[idx],'rb') as f:
                # padding with 0
                if self.padding_symbol == 0:
                    tmp = [i+1 for i in f.read()[:self.first_n_byte]]
                # padding with assigned value
                else:
                    tmp = [i for i in f.read()[:self.first_n_byte]]
                tmp = tmp + [self.padding_symbol] * (self.first_n_byte - len(tmp))

                # tmp = [i+1 for i in f.read()[:self.first_n_byte]]   # convert to [1,256], normally [0,255] 0: grey, 255: white
                # tmp = tmp+[0]*(self.first_n_byte-len(tmp))          # 0 used as padding symbol if length smaller than required
                # "padding with 256"
                # tmp = [i for i in f.read()[:self.first_n_byte]]  # convert to [0,255],  0: grey, 255: white
                # tmp = tmp + [256] * (self.first_n_byte - len(tmp))  # 256 used as padding symbol if length smaller than required

        x = np.array(tmp)
        y = np.array([self.label_list[idx]])

        return x,y



# class ImageDataset(Dataset):
#     """
#     batch loading for image data
#     load a batch of image data each time instead loading all, to reduce the workload on the memory
#
#     """
#     def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000,padding_symbol=0,image_resolution=224):
#         self.fp_list = fp_list
#         self.data_path = data_path
#         self.label_list = label_list
#         self.first_n_byte = first_n_byte
#         self.padding_symbol = padding_symbol
#         self.image_resolution = image_resolution
#
#
#     def __len__(self):
#         return len(self.fp_list)
#
#     def __getitem__(self, idx):
#         """
#         load image
#         """
#         image = Image.open(self.data_path + self.fp_list[idx]+'.png')
#         image_array = np.asarray(image)
#         image_array_flatten = list(image_array.flatten())
#         ## padding or cutting if needed
#         if len(image_array_flatten) >= self.first_n_byte:
#             image_final_array = image_array_flatten[:self.first_n_byte]
#         else:
#             image_final_array = image_array_flatten + [self.padding_symbol] * (self.first_n_byte-len(image_array_flatten))
#
#         ## reshape array to image
#         x = np.array(image_final_array)
#         x = np.reshape(x,[1,self.image_resolution,self.image_resolution])
#
#         y = np.array([self.label_list[idx]])
#
#         return x,y


class ImageDataset(Dataset):
    """
    batch loading for image data
    load a batch of image data each time instead loading all, to reduce the workload on the memory

    """
    def __init__(self,
                 fp_list=None,
                 data_path=None,
                 label_list=None,
                 transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                               transforms.ToTensor()])):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        """
        load image
        """
        x = Image.open(self.data_path + self.fp_list[idx]+'.png')
        x = self.transform(x)
        # image_array = np.asarray(image)
        # image_array_flatten = list(image_array.flatten())
        ## padding or cutting if needed
        # if len(image_array_flatten) >= self.first_n_byte:
        #     image_final_array = image_array_flatten[:self.first_n_byte]
        # else:
        #     image_final_array = image_array_flatten + [self.padding_symbol] * (self.first_n_byte-len(image_array_flatten))
        #
        # ## reshape array to image
        # x = np.array(image_final_array)
        # x = np.reshape(x,[1,self.image_resolution,self.image_resolution])

        y = np.array([self.label_list[idx]])

        return x,y


class ImageDataset_customized(Dataset):
    """
    batch loading for image data (image --> flatten --> first n byte --> reshape to image)
    load a batch of image data each time instead loading all, to reduce the workload on the memory

    """
    def __init__(self, fp_list=None,
                 data_path=None,
                 label_list=None,
                 first_n_byte=102400,
                 padding_symbol=0,
                 image_resolution=320):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte
        self.padding_symbol = padding_symbol
        self.image_resolution = image_resolution


    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        """
        load image
        """
        image = Image.open(self.data_path + self.fp_list[idx]+'.png')
        image_array = np.asarray(image)
        image_array_flatten = image_array.flatten()
        ## padding or cutting if needed
        if len(image_array_flatten) >= self.first_n_byte:
            image_final_array = image_array_flatten[:self.first_n_byte]
        else:
            image_final_array = image_array_flatten + [self.padding_symbol] * (self.first_n_byte-len(image_array_flatten))

        ## reshape array to image
        x = image_final_array / 255   ## normalized to [0,1]
        x = np.reshape(x,[1,self.image_resolution,self.image_resolution])

        y = np.array([self.label_list[idx]])

        return x,y


class ExeImageDataset(Dataset):
    """
    load binary as raw bytes and image at same time
    load a batch of data each time instead loading all, to reduce the workload on the memory

    load pre-processed image (resized to 320*320)
    """
    def __init__(self, fp_list=None, data_path=None, label_list=None, image_data_path='../data/image/',
        first_n_byte=2000000,padding_symbol=0,transform=None):
        self.fp_list = fp_list
        self.data_path = data_path  # path of all exe files
        self.image_data_path = image_data_path # path of all image files
        self.label_list = label_list
        self.first_n_byte = first_n_byte
        self.padding_symbol = padding_symbol
        self.transform = transform


    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        """
        load binary file as raw bytes and image
        """
        ## load as raw bytes
        with open(self.data_path+self.fp_list[idx],'rb') as f:
            ## padding with 0, in order to keep value in the range [0,256)
            ## the reason use 0 is because 0 won't render graident, plus 256 out of the range for bytearray
            tmp = [i for i in f.read()[:self.first_n_byte]]  # convert to [0,255],0: grey, 255: white; 256 as padding
            tmp = tmp+[self.padding_symbol]*(self.first_n_byte-len(tmp))

        ## load as image
        x_image = Image.open(self.data_path + self.fp_list[idx] + '.png')
        x_image = self.transform(x_image)

        x_byte = np.array(tmp)
        y = np.array([self.label_list[idx]])

        return (x_byte,x_image),y



# class ExeImageDataset(Dataset):
#     """
#     load binary as raw bytes and image at same time
#     load a batch of data each time instead loading all, to reduce the workload on the memory
#
#     padding with 0s eventhough 0 already used in the malware files.
#     the reason use 0 is because 0 won't render gradient, plus if use 256,
#     which is out of the range ([0,255]) for bytearray function
#     """
#     def __init__(self, fp_list=None, data_path=None, label_list=None, image_data_path='../data/image/',
#         first_n_byte=2000000,padding_symbol=0,transform=None):
#         self.fp_list = fp_list
#         self.data_path = data_path  # path of all exe files
#         self.image_data_path = image_data_path # path of all image files
#         self.label_list = label_list
#         self.first_n_byte = first_n_byte
#         self.padding_symbol = padding_symbol
#         self.transform = transform
#
#
#     def __len__(self):
#         return len(self.fp_list)
#
#     def __getitem__(self, idx):
#         """
#         load binary file as raw bytes and image
#         """
#         ## load as raw bytes
#         with open(self.data_path+self.fp_list[idx],'rb') as f:
#             ## padding with 0, in order to keep value in the range [0,256)
#             ## the reason use 0 is because 0 won't render graident, plus 256 out of the range for bytearray
#             tmp = [i for i in f.read()[:self.first_n_byte]]  # convert to [0,255],0: grey, 255: white; 256 as padding
#             tmp = tmp+[self.padding_symbol]*(self.first_n_byte-len(tmp))
#
#         ## load as image
#         x_image = Image.open(os.path.join(self.image_data_path,self.fp_list[idx])+'.png')
#         x_image = self.transform(x_image)
#
#         x_byte = np.array(tmp)
#         y = np.array([self.label_list[idx]])
#
#         return (x_byte,x_image),y









class data_normalize_inverse:
    """
    normalize data with MinMaxScaler [min_box,max_box]
    and inverse the normalized data back
    :param data: ndarray
    :return: ndarray
    """
    def __init__(self,data,min_box=0,max_box=1):

        if isinstance(data, torch.Tensor):
            data = data.data.cpu().numpy()
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        self.input_shape = data.shape
        self.input_data = data.reshape(-1,1)
        self.scaler_handle = MinMaxScaler(feature_range=(min_box,max_box))
        self.scaler = self.scaler_handle.fit(self.input_data)

    def data_normalize(self):
        data_norm = self.scaler.transform(self.input_data)
        data_norm = data_norm.reshape(self.input_shape)

        return data_norm

    def inverse_normalize(self,data_norm,output_shape=None):

        if output_shape:
            temp_shape = self.input_shape
            self.input_shape = output_shape

        input_data = data_norm.reshape(-1,1)
        data = self.scaler.inverse_transform(input_data)
        data = data.reshape(self.input_shape)

        #reset the input shape value to original value
        if output_shape:
            self.input_shape = temp_shape

        return data


# reverse-mapping from embedding value to [0,256] range
import torch
from tqdm import tqdm


def reconstruction(x, y,verbose=False):
    """
    from "https://github.com/ywkw1717/FGSM-attack-against-MalConv/blob/master/fgsm_attack.py"
    reconstruction restore original bytes from embedding matrix.
    Args:
        x torch.Tensor:
            x is word embedding (i.e. embedding result that need to recover to original byte)

        y torch.Tensor:
            y is embedding matrix (i.e. embedding dictionary)
    Returns:
        torch.Tensor:
    """
    x_size = x.size()[0]
    y_size = y.size()[0]

    z = torch.zeros(x_size)

    if verbose:
        ## 进度条显示
        for i in tqdm(range(x_size)):
            dist = torch.zeros(257)

            for j in range(y_size):
                dist[j] = torch.dist(x[i], y[j])  # computation of euclidean distance

            z[i] = dist.argmin()
    else:
        ## 无进度条显示
        for i in range(x_size):
            dist = torch.zeros(257)

            for j in range(y_size):
                dist[j] = torch.dist(x[i], y[j])  # computation of euclidean distance

            z[i] = dist.argmin()
    return z


def get_acc_FP(preds_label,original_label=None,single_model=False):
    """
    get the detection accuracy and the number of successful evasion (i.e., False Positive)
    detection accuracy: the accuracy that at least one detector identify adversarial malwares
    FP: the number of successful evasion samples, only when adversarial malware evade all detectors

    preds_label: list of predicted labels, e.g., [0,0,1,0,1,0]
    original_label: list of original labels, e.g., [1,1,1,1,1] --> all malwares
    single_model: bool. Default as False
        - True: consider results from one model
        - False: consider results from two models

    return: acc and FP for two models simutaneously, acc and FP for two model respectively
    """
    evade_success = 0
    label1,label2 = [],[]
    FP1,FP2 = 0,0
    if not single_model:
        for i, (y_1,y_2) in enumerate(preds_label):
            label1.append(y_1)
            label2.append(y_2)

            ## get FP for two models resepctively
            if y_1 == 0:
                FP1 += 1
            if y_2 == 0:
                FP2 += 1

            ## get FP/successful evasion for two models simutanously
            if y_1 == 0 and y_2 == 0:
                evade_success += 1

        ## correct_detection: at least one detector predicted as malware , label=1
        correct_detection = (i+1) - evade_success
        acc = float(correct_detection) / float(i+1)
        FP = evade_success

        ## get acc for two models resepctively
        original_label = list(np.ones(len(label1)))
        acc1 = accuracy_score(original_label,label1)
        acc2 = accuracy_score(original_label,label2)

        return acc,FP,(acc1,FP1),(acc2,FP2)
    else:
        correct_detection = sum(preds_label)
        evade_success = len(preds_label) - correct_detection
        acc = correct_detection / len(evade_success)
        FP = evade_success

        return acc,FP,(None,None),(None,None)

def get_acc_FP_3(preds_label,single_model=False):
    """
    three detectors.
    get the detection accuracy and the number of successful evasion (i.e., False Positive)
    detection accuracy: the accuracy that at least one detector identify adversarial malwares
    FP: the number of successful evasion samples, only when adversarial malware evade all detectors

    preds_label: list of predicted labels, e.g., [0,0,1,0,1,0]
    original_label: list of original labels, e.g., [1,1,1,1,1] --> all malwares
    single_model: bool. Default as False
        - True: consider results from one model
        - False: consider results from two models

    return: acc and FP for two models simutaneously, acc and FP for two model respectively
    """
    evade_success = 0
    label1,label2,label3 = [],[],[]
    FP1,FP2,FP3 = 0,0,0
    if not single_model:
        for i, (y_1,y_2,y_3) in enumerate(preds_label):
            label1.append(y_1)
            label2.append(y_2)
            label3.append(y_3)

            ## get FP for two models resepctively
            if y_1 == 0:
                FP1 += 1
            if y_2 == 0:
                FP2 += 1
            if y_3 == 0:
                FP3 += 1

            ## get FP/successful evasion for three models simutanously
            if y_1 == 0 and y_2 == 0 and y_3==0:
                evade_success += 1

        ## correct_detection: at least one detector predicted as malware , label=1
        correct_detection = (i+1) - evade_success
        acc = float(correct_detection) / float(i+1)
        FP = evade_success

        ## get acc for two models resepctively
        original_label = list(np.ones(len(label1)))
        acc1 = accuracy_score(original_label,label1)
        acc2 = accuracy_score(original_label,label2)
        acc3 = accuracy_score(original_label,label3)

        return acc,FP,(acc1,FP1),(acc2,FP2),(acc3,FP3)
    else:
        correct_detection = sum(preds_label)
        evade_success = len(preds_label) - correct_detection
        acc = correct_detection / len(evade_success)
        FP = evade_success

        return acc,FP,(None,None),(None,None),(None,None)


class forward_prediction_process:
    """
    integrate several sub-functions, e.g., forward, prediction, get_sign, etc
    """
    def __init__(self):
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


    def _forward(self,input, model,model_type='byte'):
        """
        forward process of NN model to get predictions
        input: [N,W] <==> [batch,width] should be long format since it feed into embed layer first
        model: target model

        return:
            - predictions, 1-D array
            - embeded input after embed layer

        note: "_" in the name of function indicates that it won't be import by "import *"
        """
        if model_type == 'byte':
            embed = model.embed
            embed_x = embed(input.long()).detach()
        else:
            embed_x = input
        embed_x.requires_grad = True
        output = model(embed_x.to(self.device))
        if not output.shape == (1,2):
            output=torch.unsqueeze(output,dim=0)
        preds = F.softmax(output,dim=1)
        preds = preds.cpu().detach().numpy().squeeze()

        return preds, embed_x


def get_adv_name(partial_dos:str='False',
                 content_shift:str='False',
                 slack:str='False',
                 combine_w_slack:str='False',
                 preferable_extension_amount:int=0,
                 preferable_shift_amount:int=0):
    """
    get and return the adversarial evasion attack name
    :param partial_dos:
    :param content_shift:
    :param slack:
    :param combine_w_slack:
    :param preferable_extension_amount:
    :param preferable_shift_amount:
    :return:
    """
    if partial_dos == 'True' and content_shift == 'False' and slack == 'False':
        return 'partialDos'
    elif partial_dos == 'False' and content_shift == 'True' and slack == 'False':
        return 'contentShift'
    elif partial_dos == 'False' and content_shift == 'False' and slack == 'True':
        return 'slack'
    elif partial_dos == 'True' and content_shift == 'True' and slack == 'False':
       return 'partialDosContentShift'
    elif partial_dos == 'True' and content_shift == 'False' and slack == 'True':
        return 'partialDosSlack'
    elif partial_dos == 'False' and content_shift == 'True' and slack == 'True':
        return 'contentShiftSlack'
    elif partial_dos == 'True' and content_shift == 'True' and slack == 'True':
        return 'partialDosContentShiftSlack'
    else:
        if preferable_shift_amount == 0 and combine_w_slack == 'False':
            if preferable_extension_amount == 0:
                return 'fullDos'
            else:
                return 'DosExtension'
        elif preferable_shift_amount > 0 and combine_w_slack == 'False':
            if preferable_extension_amount == 0:
                return 'FullDosContentShift'
            else:
                return 'DosExtensionContentShift'
        elif preferable_shift_amount == 0 and combine_w_slack == 'True':
            if preferable_extension_amount == 0:
                return 'FullDosSlack'
            else:
                return 'DosExtensionSlack'
        elif preferable_shift_amount > 0 and combine_w_slack == 'True':
            if preferable_extension_amount == 0:
                return 'FullDosContentShiftSlack'
            else:
                return 'DosExtensionContentShiftSlack'