from pickletools import markobject
from turtle import color
from matplotlib import pyplot as plt
import json
import os
import seaborn as sns
import numpy as np

def main():
     generate_rank_experiments_plot()

def generate_rank_experiments_plot():
     val_svd = []
     val_nmf = []
     val_als = []

     fs = os.listdir("./log/")
     for f in fs:
          if f[:3] == "SVD":
               with open("./log/" + f) as json_file:
                    js = json.load(json_file)
                    val_svd.append([js["parameters"]["rank"], js["val_rmse"][0]])

          elif f[:3] == "NMF":
               with open("./log/" + f) as json_file:
                    js = json.load(json_file)
                    val_nmf.append([js["parameters"]["rank"], js["val_rmse"][0]])

          elif f[:3] == "ALS":
               with open("./log/" + f) as json_file:
                    js = json.load(json_file)
                    val_als.append([js["parameters"]["rank"], min(js["val_rmse"])])

     
     #    train = sorted(train, key=lambda x: x[0])
     val_nmf = sorted(val_nmf, key=lambda x: x[0])
     val_svd = sorted(val_svd, key=lambda x: x[0])
     val_als = sorted(val_als, key=lambda x: x[0])
     
     sns.set_style("white")
     plt.plot([t[0] for t in val_svd], [t[1] for t in val_svd], '-x', label = 'SVD', markevery=[5])
     plt.plot([t[0] for t in val_nmf], [t[1] for t in val_nmf], '-x', label = 'NMF', markevery=[9])
     plt.plot([t[0] for t in val_als], [t[1] for t in val_als], '-x', label = 'ALS', markevery=[1])
     plt.annotate("r=7", (6.4, 1.01))
     plt.annotate("r=11", (10.4, 1.01))
     plt.annotate("r=3", (2.4, 0.9845))
     
     plt.ylim(top = 1.02, bottom=0.983)

     plt.ylabel('Validation RMSE', fontsize = 12)
     plt.xlabel('rank', fontsize = 12)
     plt.legend()
     plt.savefig("./plots/rank_analysis.png")



if __name__ == '__main__':
    main()