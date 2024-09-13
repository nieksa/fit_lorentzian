import matplotlib.pyplot as plt
import torch
import numpy as np
from generator import generator_par, complex_split_lorentzian

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

    # 把 prediction 的参数带入洛伦兹方程计算拟合。

    y_prediction = complex_split_lorentzian(prediction.ravel(),x)
    # print(y_prediction)
    # 绘图示例
    # 假设你想绘制预测结果与原始数据的关系
    plt.figure(figsize=(10, 5))
    plt.plot(plot_y, label='Input Data')
    plt.plot(y_prediction, label='Model Prediction', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Model Prediction vs. Input Data')
    plt.show()