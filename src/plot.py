import pandas as pd
import matplotlib.pyplot as plt

file = "log/immuno_train_log.csv"
file = "log/bind_train_log.csv"

if file == "log/bind_train_log.csv":
    data = pd.read_csv('bind_train_log.csv')
    epochs = data['epoch']
    train_loss = data['loss']
    train_mse = data['mse']
    val_loss = data['val_loss']
    val_mse = data['val_mse']

    # 绘制训练损失和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # 绘制训练MSE和验证MSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mse, label='Training MSE')
    plt.plot(epochs, val_mse, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training and Validation MSE')
    plt.show()

    # 绘制训练损失和MSE对比曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, train_mse, label='Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/MSE')
    plt.legend()
    plt.title('Training Loss and MSE')
    plt.show()

    # 绘制验证损失和MSE对比曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.plot(epochs, val_mse, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/MSE')
    plt.legend()
    plt.title('Validation Loss and MSE')
    plt.show()


if file == "log/immuno_train_log.csv":
    data = pd.read_csv(file)
    epochs = data['epoch']
    train_accuracy = data['categorical_accuracy']
    train_loss = data['loss']
    val_accuracy = data['val_categorical_accuracy']
    val_loss = data['val_loss']

    # 绘制训练集和验证集上的分类准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    # 绘制训练集和验证集上的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
