import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

def plot_loss_and_mse(epochs, train_loss, val_loss, train_mse, val_mse):    
    # 绘制训练MSE和验证MSE曲线
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_mse, label='Training MSE')
    plt.plot(epochs, val_mse, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training and Validation MSE')

    plt.savefig("plot/bind_MSE_plot.png", dpi=300)
    
    # plt.show()

def plot_accuracy_and_loss(epochs, train_accuracy, val_accuracy, train_loss, val_loss):
    # 绘制训练集和验证集上的分类准确率曲线
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.savefig("plot/immuno_loss_plot.png", dpi=300)
    
    # 绘制训练集和验证集上的损失曲线
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.savefig("plot/immuno_accuracy_plot.png", dpi=300)

    # plt.show()

def main():
    # 文件名定义为常量
    BIND_LOG_FILE = "log/bind_train_log.csv"
    IMMUNO_LOG_FILE = "log/immuno_train_log.csv"

    file = BIND_LOG_FILE

    if file == BIND_LOG_FILE:
        data = pd.read_csv(file)
        epochs = data['epoch']
        train_loss = data['loss']
        train_mse = data['mse']
        val_loss = data['val_loss']
        val_mse = data['val_mse']

        plot_loss_and_mse(epochs, train_loss, val_loss, train_mse, val_mse)


    file = IMMUNO_LOG_FILE

    if file == IMMUNO_LOG_FILE:
        data = pd.read_csv(file)
        epochs = data['epoch']
        train_accuracy = data['categorical_accuracy']
        train_loss = data['loss']
        val_accuracy = data['val_categorical_accuracy']
        val_loss = data['val_loss']

        plot_accuracy_and_loss(epochs, train_accuracy, val_accuracy, train_loss, val_loss)

main()
