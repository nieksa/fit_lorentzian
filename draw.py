import matplotlib.pyplot as plt
import torch
import numpy as np
from generator import generator_par, complex_split_lorentzian
from sklearn.metrics import mean_squared_error

def draw_prediction(model):
    x = np.linspace(-200, 200, 400)
    plot_param = generator_par()
    plot_y = complex_split_lorentzian(plot_param,x)
    print("真实参数:", plot_param)

    # 使用模型进行预测
    with torch.no_grad():
        model.eval()  # 确保模型在评估模式
        prediction = model(torch.tensor(plot_y,dtype=torch.float32)).numpy()
    print("预测结果:", prediction)
    y_prediction = complex_split_lorentzian(prediction.ravel(),x)
    mse = mean_squared_error(plot_y, y_prediction)
    print("均方误差:",mse)
    # 计算 y_prediction 与 plot_y 之间的误差。选取一种合适的数学表达
    plt.figure(figsize=(10, 5))
    plt.plot(x, plot_y, label='Input Data')  # 确保使用正确的 x 值
    plt.plot(x, y_prediction, label='Model Prediction', linestyle='--')
    plt.xlabel('Frequency')

    plt.ylabel('Value')
    plt.legend()
    plt.title('Model Prediction vs. Input Data')
    plt.text(0.05, 0.35, f'real: {plot_param}', transform=plt.gca().transAxes, fontsize=9)
    plt.text(0.05, 0.3, f'pred: {prediction}', transform=plt.gca().transAxes, fontsize=9)
    plt.text(0.05, 0.5, f'mse: {mse}', transform=plt.gca().transAxes, fontsize=9)
    plt.show()