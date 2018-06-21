import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

dat = pd.DataFrame([(-0.86, 5, 0),(-0.3, 5, 1),(-0.05, 5, 3),(0.73, 5, 5)], 
                    columns = ['dose','animal_num','death_num'])
dat['ratio'] = dat.death_num / dat.animal_num
dat['logit_ratio'] = log


dat = [(-0.86, 5, 0),(-0.3, 5, 1),(-0.05, 5, 3),(0.73, 5, 5)]

def dat_create_p1(dose, animal_num, death_num):
    x = np.repeat(dose, animal_num)
    y = np.append(np.repeat(1, death_num), np.repeat(0, animal_num - death_num))
    return (x,y)

def dat_create_p2(data):
    x, y = np.empty(1), np.empty(1, dtype = int)
    for dose, ani_num, dea_num in data:
        x_p, y_p = dat_create_p1(dose,ani_num, dea_num)
        x, y = np.append(x, x_p), np.append(y, y_p)
    x, y = x[1:], y[1:]
    return (x,y)

new_dat = dat_create_p2 (dat)


lr = LogisticRegression()
lr.fit(new_dat[0].reshape(-1,1), new_dat[1])
print(lr.coef_)
print(lr.intercept_)
