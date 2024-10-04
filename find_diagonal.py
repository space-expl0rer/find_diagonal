
#TODO:
#DONE visualize gradient, make visualization as a function scalable for big nns
#DONE print loss surface make visualization applicable for big nns
#MNISTexperiment with dynamic stepsize
#DONE вынести плотника в отдельную либу
#MNISTexperiment with batchsize and epoch size
#MNISTtry firstly train many times on first, only then on second example
#DONEconvert everything to matrix multiplication
#MNISTmeasure perfomance
#MNISTanalyze the similarity of 5's in the dataset
#MNISTtry to compress data somehow to get better results
#MNISTexperiment with normalization
#MNISTtry to add and maximize margin
#MNISTtry l1 loss
#MNISTadd evaluation
#TODO добавить русские комментарии в код сюда и в плотник и сабмитить на гитхаб
#TODO назвать проект выжимаем всё из линейного классификатора на мнист (и спидап сделать)
import numpy as np
import matplotlib.pyplot as plt
from Plotnik import Plotnik

'''
Попробуем заставить нейрон выучить простейший шаблон на выдуманном датасете из 16 картинок.
Посмотрим, получится ли у "сети" понять, что искомый паттерн - это диагональ.
Визуализацию вынес в отдельный класс (и файл) plotnik для компактности кода.
'''

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

#создаём 1 нейрон и датасет (с bias)
rng = np.random.default_rng()
network = rng.uniform(-1, 1, 5)

dataset = np.array([list(i) + [1] for i in product([0, 1], repeat=4)])

labels = np.array([i[0] and i[3] for i in dataset])
#белый - единицы, черный - нули
fig, axes = plt.subplots(4, 4)
for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(np.reshape(dataset[i][:-1], (2, 2)), cmap='grey')
    ax.set_title(labels[i])
plt.show()

#тут нужна сигмоида, тк преобразовываем сигнал в диапазон от 0 до 1
epoch_num = 256
stepsize = 0.1
loss_v = []
#используем plotnik для отрисовки графиков и таблиц с весами и градиентами
plotter = Plotnik()
for epoch in range(epoch_num):
    #рассчитываем predсказание сразу по всем примерам
    pred = 1 / (1 + np.exp(-1 * np.dot(dataset, network.T)))
    #l2 loss = (pred - ans) ** 2
    #поэтому производная = 2 * (pred - ans) * pred * (1 - pred)
    loss_v.append(sum((pred - labels) ** 2))

    #считаем градиент по всем сразу
    grad = np.dot(2 * (pred - labels) * pred * (1 - pred), dataset)
    network -= grad * stepsize
    #посмотрим на срез поверхности лосса по двум весам
    arr_w0 = []
    arr_w1 = []
    arr_l = []
    w0_real = network[0] #запомним и потом вернём на место
    w1_real = network[1] #тоже
    w0 = w0_real - 5

    while w0 < w0_real + 5:
        arr_w0.append(w0)
        w1 = w1_real - 5
        network[0] = w0
        while w1 < w1_real + 5:
            network[1] = w1
            pred = 1 / (1 + np.exp(-1 * np.dot(dataset, network.T)))
            arr_l.append(sum((pred - labels) ** 2))
            w1 += 0.1
        w0 += 0.1
    w1 = w1_real - 5
    while w1 < w1_real + 5:
        arr_w1.append(w1)
        w1 += 0.1

    network[0] = w0_real
    network[1] = w1_real
    arr_w0 = np.array(arr_w0)
    arr_w1 = np.array(arr_w1)
    arr_l = np.reshape(np.array(arr_l), (len(arr_w0), len(arr_w1)))
    #и красиво всё выводим
    plotter.update(epoch, loss_values=loss_v,
                   weights=network, gradients=grad,
                   weight_1=arr_w0, weight_2=arr_w1,
                   loss_surface=arr_l, w1_num=0, w2_num=1,
                   text_info=f'stepsize = {stepsize}')
#это чтобы рисунки не исчезали по выходу из цикла
plotter.remain()

#посмотрим что там предсказывает нейрон
for i in dataset:
    print(i, 1 / (1 + np.exp(-1 * np.dot(network, i))), i[0] and i[3])

'''
Отлично. После обучения веса сети, соответствующие  главной диагонали положительны,
а веса побочной диагонали наоборот слегка ушли в минус.
'''
