import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


class data_normalize_inverse:
    """
    normalize data with MinMaxScaler [0,1]
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

    def inverse_normalize(self,data_norm):
        input_data = data_norm.reshape(-1,1)
        data = self.scaler.inverse_transform(input_data)
        data = data.reshape(self.input_shape)

        return data

data = np.array([1,2,3])
normalizer = data_normalize_inverse(data,0,1)
norm = normalizer.data_normalize()
rever = normalizer.inverse_normalize(norm+[0,0.6,0])
print(f'normiza data {norm}, {type(norm)}')
print(f'inversed data {rever}')
