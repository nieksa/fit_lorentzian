import numpy as np
def generator_par(normal_distribution):
    k0_lim = [5, 50]
    kEx_lim = [1, 50]
    w0_lim = [-20, 20]
    kCr_lim = [0, 10]
    kCi_lim = [-100, 100]


    if normal_distribution:
        k0_mean, k0_std = np.mean(k0_lim), (k0_lim[1] - k0_lim[0]) / 2
        kEx_mean, kEx_std = np.mean(kEx_lim), (kEx_lim[1] - kEx_lim[0]) / 2
        w0_mean, w0_std = np.mean(w0_lim), (w0_lim[1] - w0_lim[0]) / 2
        kCr_mean, kCr_std = np.mean(kCr_lim), (kCr_lim[1] - kCr_lim[0]) / 2
        kCi_mean, kCi_std = np.mean(kCi_lim), (kCi_lim[1] - kCi_lim[0]) / 2
        params = np.array([
            np.random.normal(k0_mean, k0_std),
            np.random.normal(kEx_mean, kEx_std),
            np.random.normal(w0_mean, w0_std),
            np.random.normal(kCr_mean, kCr_std),
            np.random.normal(kCi_mean, kCi_std)
        ])
        params[0] = np.clip(params[0], k0_lim[0], k0_lim[1])
        params[1] = np.clip(params[1], kEx_lim[0], kEx_lim[1])
        params[2] = np.clip(params[2], w0_lim[0], w0_lim[1])
        params[3] = np.clip(params[3], kCr_lim[0], kCr_lim[1])
        params[4] = np.clip(params[4], kCi_lim[0], kCi_lim[1])
        return params
    params = np.random.uniform(low=[k0_lim[0], kEx_lim[0], w0_lim[0], kCr_lim[0], kCi_lim[0]],
                              high=[k0_lim[1], kEx_lim[1], w0_lim[1], kCr_lim[1], kCi_lim[1]], 
                              size=(5,))
    return params


def complex_split_lorentzian(par, x):
    w0 = par[0]
    k0 = par[1]
    kEx = par[2]
    kCR = par[3]
    kCI = par[4]
    
    term1 = 1j * (x - w0) + (k0 + kEx) / 2
    term2 = -kCR + 1j * kCI
    denominator = term1**2 - (term2 / 2)**2
    y = np.abs(1 - (kEx * term1 / denominator))**2
    
    return y

def generate_data(nums, normal_distribution = False):

    x = np.linspace(-200, 200, 400)  # 定义 x 范围与采样点

    x_data = []
    y_data = []
    for _ in range(nums):
        param = generator_par(normal_distribution)
        y = complex_split_lorentzian(param, x)

        if np.sum(y) > 0:
            x_data.append(param)
            y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data