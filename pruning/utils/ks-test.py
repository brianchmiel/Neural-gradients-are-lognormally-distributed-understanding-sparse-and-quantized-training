



import numpy as np
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
K  = 30000

fig, axs = plt.subplots(7,7, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

Laplace =[]
Laplace_P =[]
Normal =[]
Normal_P =[]
Logistic =[]
Logistic_P =[]
Cauchy =[]
Cauchy_P =[]
Uniform =[]
Uniform_P =[]
Loglaplace =[]
Loglaplace_P =[]
Lognorm =[]
Lognorm_P =[]

i=0
# for subdir, dirs, files in os.walk('C:/Users/rbanner/Desktop/dumps_maxim/full_model/epoch90'):
# for subdir, dirs, files in os.walk('C:/Users/rbanner/Desktop/dumps_maxim/full_model/weight_epoch90'):
for subdir, dirs, files in os.walk('C:/Users/rbanner/Desktop/YuryDumps'):
    for file in files:
        i = i + 1
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        tensor = np.random.choice(np.load(filepath).flatten(),K)
        # Tensor = stats.norm.rvs(loc=0, scale =1, size=100000)
        Tensor = (tensor - np.mean(tensor)) / np.std(tensor)

        Laplace.append(stats.kstest(Tensor, 'laplace')[0])
        Laplace_P.append(stats.kstest(Tensor, 'laplace')[1])
        Normal.append(stats.kstest(Tensor, 'norm')[0])
        Normal_P.append(stats.kstest(Tensor, 'norm')[1])
        Uniform.append(stats.kstest(Tensor, 'uniform')[0])
        Uniform_P.append(stats.kstest(Tensor, 'uniform')[1])
        Logistic.append(stats.kstest(Tensor, 'logistic', args=(0, 1))[0])
        Logistic_P.append(stats.kstest(Tensor, 'logistic', args=(0, 1))[1])
        Cauchy.append(stats.kstest(Tensor, 'cauchy', args=(0, 1))[0])
        Cauchy_P.append(stats.kstest(Tensor, 'cauchy', args=(0, 1))[1])
        Loglaplace.append(stats.kstest(Tensor, 'loglaplace', args=(1, 0))[0])
        Loglaplace_P.append(stats.kstest(Tensor, 'loglaplace', args=(1, 0))[1])
        Lognorm.append(stats.kstest(Tensor, 'lognorm', args=(1, 0))[0])
        Lognorm_P.append(stats.kstest(Tensor, 'lognorm', args=(1, 0))[1])
        print('a')

    print('laplace: D:' + str(np.mean(Laplace))+ 'p_value=' + str(np.mean(Laplace_P)))
    print('normal: D' +  str(np.mean(Normal))+ 'p_value=' + str(np.mean(Normal_P)))
    print('uniform: D' +  str(np.mean(Uniform))+ 'p_value=' + str(np.mean(Uniform_P)))
    print('loglaplace : ' +  str(np.mean(Loglaplace))+ 'p_value=' + str(np.mean(Loglaplace_P)))
    print('lognorm : ' +  str(np.mean(Lognorm))+ 'p_value=' + str(np.mean(Lognorm_P)))
    print('Cauchy  : ' +  str(np.mean(Cauchy))+ 'p_value=' + str(np.mean(Cauchy_P)))
    print('logistic  : ' +  str(np.mean(Logistic))+ 'p_value=' + str(np.mean(Logistic_P)))

# print('laplace: ' + str(stats.kstest(Tensor, 'laplace')))
# print('normal: ' + str(stats.kstest(Tensor, 'norm')))
# print('uniform: ' + str(stats.kstest(Tensor, 'uniform')))
# print('loglaplace : ' + str(stats.kstest(Tensor, 'loglaplace', args=(1, 0))))
# print('lognorm : ' + str(stats.kstest(Tensor, 'lognorm', args=(1, 0))))
# print('Cauchy  : ' + str(stats.kstest(Tensor, 'cauchy', args=(0, 1))))
# print('logistic  : ' + str(stats.kstest(Tensor, 'logistic', args=(0, 1))))

# print('cosine: ' + str(stats.kstest(Tensor, 'cosine', args=(0, 1))))
