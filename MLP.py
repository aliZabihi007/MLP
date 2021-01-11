import pandas as pd
import numpy as np
import random
import progressbar
import sys
import matplotlib.pyplot as plt


# 808
# 80.80000000000001

class MLP:
    def __init__(self, x, y):
        self.w1 = np.array(np.zeros((24, 12)))
        self.w2 = np.array(np.zeros((12, 1)))
        self.hidden = np.array(np.zeros((12)))
        self.biase = np.array(np.zeros((13)))
        self.X = x
        self.y = y
        self.count = 0
        self.lastoutput = np.array(np.zeros((5)))
        self.gradiantoutput = np.array(np.zeros((13)))

    # تعریف تابعی برای مقدار دهی اولیه کردن به وزن ها و بایاس
    def createdataarray(self):
        for i in range(0, 24, 1):
            for j in range(0, 12, 1):
                self.w1[i][j] = round(random.uniform(0, 1), 4)
        print("self.w1")
        print(self.w1)
        for i in range(0, 12, 1):
            self.w2[i][0] = round(random.uniform(0, 1), 4)
        print("self.w2")
        print(self.w2)
        for i in range(0, 13, 1):
            self.biase[i] = round(random.uniform(0, 1), 4)

    # تعریف تابعی برای حرکت کردن به سمت جلو و ضرب و جمع مقاادیر وزن و بدست اوردن خروجی مطلوب

    def forwardingPass(self, indexX) -> float:
        self.hidden = np.dot(self.X[indexX], self.w1)

        for i in range(0, 12, 1):
            self.hidden[i] = self.sigmoid(self.hidden[i] + self.biase[i])

        result = np.sum(np.dot(self.hidden, self.w2))
        return self.sigmoid(result + self.biase[12])

    #  ارزشیابی میکنیم که مقدار خروجی و مقدار جوابی که خودمان داریم چه جوابی بهینه خواهد بود
    def evaluationMLP(self, output, indexY, teta) -> bool:
        if (np.power(self.y[indexY] - output, 2) / 2 < teta):

            return True
        # self.lastoutput[indexY] = output
        else:
            return False

    # آبدیت کردن وزن ها و اصلاح وزن ها و بدست اوردن مقادیر مطلوب
    def backpropagation(self, indexY, outy, learningrate):
        self.gradiantoutput[0] = outy * (1 - outy) * (self.y[indexY] - outy)
        for i in range(1, 13, 1):
            self.gradiantoutput[i] = self.hidden[i - 1] * (1 - self.hidden[i - 1]) * (
                    self.gradiantoutput[0] * self.w2[i - 1])
        self.updateYtoHidden(learningrate)
        self.updateHiddenToX(learningrate, indexY)

    # ابدیت کردن مقدار وزن
    def updateYtoHidden(self, eta):

        for i in range(0, 12, 1):
            self.w2[i][0] = self.w2[i][0] + (eta * self.gradiantoutput[0] * self.hidden[i])
        self.biase[12] = self.biase[12] + (1 * self.gradiantoutput[0] * eta)

    # ابدیت کردن وزن در مرحله دوم
    def updateHiddenToX(self, eta, indexX):
        for i in range(0, 12, 1):
            for j in range(0, 24, 1):
                self.w1[j][i] = self.w1[j][i] + (eta * self.gradiantoutput[i + 1] * self.X[indexX][j])
            self.biase[i] = self.biase[i] + (1 * self.gradiantoutput[i + 1] * eta)

    # تعریف تابع سیگموید
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # نمایش دادن خروجی مطلوب
    def showReasult(self):
        sys.stdout.write("\033[1;31m")
        print("weight x TO hidden")
        print(self.w1)
        sys.stdout.write("\033[95m")
        print("weight hidden TO y")
        print(self.w2)
        sys.stdout.write("\033[94m")

    def predict(self, y_t, x_t):
        boolT = 0
        for j in range(0, 1000, 1):
            self.hidden = np.dot(x_t[j], self.w1)

            for i in range(0, 12, 1):
                self.hidden[i] = self.sigmoid(self.hidden[i] + self.biase[i])

            result = np.sum(np.dot(self.hidden, self.w2))
            out = self.sigmoid(result + self.biase[12])

            if (np.power(y_t[j] - out, 2) / 2 < 0.1):
                boolT += 1

        print(boolT)
        print((boolT / 1000) * 100)


if __name__ == '__main__':

    df = pd.read_csv('Churn Data.csv', sep=',')
    target = list()
    for item in df.iloc:
        target.append(item['Exited'])

    # print(target)
    demmyG = pd.get_dummies(df['Geography'])
    demmyGE = pd.get_dummies(df['Gender'])
    demmyTen = pd.get_dummies(df['Tenure'])
    demmyNumOfP = pd.get_dummies(df['NumOfProducts'])
    demmyHasCrC = pd.get_dummies(df['HasCrCard'])
    demmyIsActi = pd.get_dummies(df['IsActiveMember'])
    finall = pd.concat((demmyG, demmyGE, demmyTen, demmyNumOfP, demmyHasCrC, demmyIsActi), axis=1)
    enter = finall.to_numpy()
    #جداسازی دیتا های اموزش و تست 
    x_train = enter[0:8000]
    y_trian = target[0:8000]
    x_test = enter[8000:9000]
    y_test = target[8000:9000]
    mlp = MLP(x_train, y_trian)
     #مقدار دهی اولیه یه وزن ها 
    mlp.createdataarray()
    epak = 0
    print("wainting .....")
    cont = 0
    contlist = list()
    while (epak < 10):
        cont = 0
        for i in range(0, 8000, 1):
            out = mlp.forwardingPass(i)
            BOOL = mlp.evaluationMLP(out, i, 0.01)
            if (BOOL == False):
                mlp.backpropagation(i, out, 0.1)
            else:
                cont += 1
        epak = epak + 1
        contlist.append((cont * 100) / 8000)
        print(cont )
    plt.plot(contlist)
    plt.show()
    mlp.showReasult()
    mlp.predict(y_test, x_test)
