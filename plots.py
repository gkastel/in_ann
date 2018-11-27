import numpy
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import scipy.stats

ar_sub = numpy.loadtxt('errors-sub.txt',  dtype=float)
ar_supra = numpy.loadtxt('errors-supra.txt', dtype=float)
ar_linear = numpy.loadtxt('errors-linear.txt', dtype=float)
ar_mixed = numpy.loadtxt('errors-mixed.txt', dtype=float)


if True:
    plt.figure();
    plt.plot(ar_sub[:, 1])
    plt.plot(ar_supra[:, 1])
    plt.plot(ar_linear[:, 1])
    plt.plot(ar_mixed[:, 1])
    plt.title("Test set Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Error")
    plt.legend(["Sub","Supra", "Linear", "Mixed"] )



if True:
    plt.figure()
    plt.plot(ar_sub[:, 0])
    plt.plot(ar_supra[:, 0])
    plt.plot(ar_linear[:, 0])
    plt.plot(ar_mixed[:, 0])
    plt.title("Training Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Error")
    plt.legend(["Sub","Supra", "Linear", "Mixed"] )




# Plot predicted
if True:
    for fig in ['sub','supra','mixed','linear']:
        preds   = numpy.loadtxt('predictions-%s.txt'%(fig),  dtype=float)
        actual  = numpy.loadtxt('actual-%s.txt'%(fig),  dtype=float)
        plt.figure()
        plt.scatter(preds,actual, marker='.')
        pearsonr = scipy.stats.pearsonr(preds, actual)
        print(pearsonr[0])
        print("R= %f"%(pearsonr[0]))
        plt.title("%s ,  r=%f"%( fig,  pearsonr[0]))


plt.show()
