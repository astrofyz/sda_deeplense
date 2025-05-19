from util_module import *

def lognorm_old(data):
    data_log = np.log10(data + 1e-10)
    data_max = data_log.max(axis=(-1, -2))
    data_min = data_log.min(axis=(-1, -2))
    data_norm = (data_log - data_min[:, :, :, np.newaxis, np.newaxis])/(data_max[:, :, :, np.newaxis, np.newaxis] - data_min[:, :, :, np.newaxis, np.newaxis])
    data_norm = np.nan_to_num(data_norm)
#     print("called log")
    return data_norm

def lognorm(data):
    masked_data = np.ma.masked_equal(data, 0.)
    data_log = np.log10(masked_data)
    data_max = data_log.max(axis=(-1, -2))
    data_min = data_log.min(axis=(-1, -2))
    data_norm = (data_log - data_min[:, :, :, np.newaxis, np.newaxis])/(data_max[:, :, :, np.newaxis, np.newaxis] - data_min[:, :, :, np.newaxis, np.newaxis])
#     data_norm = np.nan_to_num(data_norm)
#     print("called log")
    data_norm[np.where(data == 0.)] = 0.
    return data_norm.data

def sqrtnorm(data):
#     print("called sqrt")
    return np.sqrt(data)


def idnorm(data):
#     print("called idnorm")
    return data


def loadimg(path):
    assert isinstance(path, str)
    if path.endswith('.fits'):
        datafits = fits.open(path)
        if len(datafits) == 1:
            return datafits[0].data
        else:
            return fits.open(path)[1].data  # for HSC lenses data is in 1 element
    elif path.endswith('.npy'):
        return np.load(path)
        

def makeimg(path):
    # print(path)
    if isinstance(path, np.ndarray):
        assert len(path) == 3
        data_0 = loadimg(path[0])
        data_1 = loadimg(path[1])
        data_2 = loadimg(path[2])
        data = np.concatenate([data_0, data_1, data_2])
        data = data.reshape(3, -1, data.shape[1])
        return torch.from_numpy(data)       
    assert isinstance(path, str)
    data = loadimg(path)
    return torch.from_numpy(data.astype(np.float32))


def preprocess(img):
    assert isinstance(img, torch.Tensor)
    # print("Preprocess", img.shape, img.max(), img.min(), img.mean(), img.std())
    img -= img.amin((1,2), keepdim=True)
    img /= img.amax((1, 2), keepdim=True)
    img = torch.nan_to_num(img, 0.)
    # print("Preprocessed", img.max(), img.min(), img.mean(), img.std())
#    img = np.sqrt(img)  # try sqrt
    return img


func_dict = {"none": idnorm, "sqrt": sqrtnorm, "log": lognorm}
