from pickletools import markobject
from turtle import color
from matplotlib import pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns
import numpy as np

def main():
     generate_rank_experiments_plot()
     generate_ensemble_weights_plot()

def generate_ensemble_weights_plot():
     test = pd.read_csv('log/ensemble/test_true.csv')['Prediction']
     bfm_preds = pd.read_csv('log/ensemble/bfm_preds.csv')['Prediction']
     als_preds = pd.read_csv('log/ensemble/als_preds.csv')['Prediction']
     sim_preds = {
          'none_30' : pd.read_csv('log/ensemble/sim_preds_w_none_n_30.csv')['Prediction'],
          'none_10000' : pd.read_csv('log/ensemble/sim_preds_w_none_n_10000.csv')['Prediction'],
          'normal_30' : pd.read_csv('log/ensemble/sim_preds_w_normal_n_30.csv')['Prediction'],
          'normal_10000' : pd.read_csv('log/ensemble/sim_preds_w_normal_n_10000.csv')['Prediction']
     }

     sim_options_to_label = {
          'none_30' : 'no weighting, 30 neighbors',
          'none_10000' : 'no weighting, all neighbors',
          'normal_30' : ' normal weighting, 30 neighbors',
          'normal_10000' : 'normal_weighting, all neighbors'
     }

     # Best RMSE: 0.9697030734079068
     # 100*BFM + 19*Sim

     for i in range(40):
          weights = {'bfm': 100, 'sim': i}
          bfm_sim_preds = (np.array(bfm_preds) * weights['bfm'] + np.array(sim_preds['normal_30']) * weights['sim']) / sum(weights.values())
          val_by_weight = {}
          for j in range(10):
               weights = {'bfm_sim': 100, 'als': j}

               weighted_preds = (np.array(bfm_sim_preds) * weights['bfm_sim'] + np.array(als_preds) * weights['als']) / sum(weights.values())
               rmse = ((np.array(test) - np.array(weighted_preds)) ** 2).mean() ** .5
               val_by_weight[j] = rmse

          lists = sorted(val_by_weight.items())
          x, y = zip(*lists)
          plt.plot(x, y, label = 'Sim = ' + str(i) + '% of BFM')
     plt.ylabel('Validation RMSE', fontsize = 12)
     plt.xlabel('ALS weight (BFM+Sim having weight 100)', fontsize = 12)
     plt.legend()
     plt.savefig("./plots/BFM_Sim_ALS_weights.png")
     plt.cla()

     val_by_weight = {}
     for model_description, pred in sim_preds.items():
          for i in range(0, 50, 1):
               weights = {'bfm': 100, 'sim': i}

               weighted_preds = (np.array(bfm_preds) * weights['bfm'] + np.array(pred) * weights['sim']) / sum(weights.values())
               rmse = ((np.array(test) - np.array(weighted_preds)) ** 2).mean() ** .5
               val_by_weight[i] = rmse

          lists = sorted(val_by_weight.items())
          x, y = zip(*lists)
          plt.plot(x, y, label = 'Validation error')
          plt.ylabel('Validation RMSE', fontsize = 12)
          plt.xlabel('Similarity weight (BFM having weight 100)', fontsize = 12)
          plt.legend()
          plt.savefig("./plots/BFM_Sim_" + model_description + "_weights.png")
          plt.cla()

def generate_rank_experiments_plot():
     val_svd = []
     val_nmf = []
     val_als = []
     val_funk = []
     val_bfm = []
     val_bfm_iters = []
     val_bfm_options_rank = {
          '_______': [],
          '___iu__': [],
          '_____ii': [],
          '___iuii': [],
          'ord____': [],
          'ordiu__': [],
          'ord__ii': [],
          'ordiuii': []
     }
     val_bfm_options_iters = {
          '_______': [],
          '___iu__': [],
          '_____ii': [],
          '___iuii': [],
          'ord____': [],
          'ordiu__': [],
          'ord__ii': [],
          'ordiuii': []
     }

     options_to_label = {
          '_______': 'BFM',
          '___iu__': 'BFM with implicit user information',
          '_____ii': 'BFM with implicit movie information',
          '___iuii': 'BFM with implicit user and movie information',
          'ord____': 'BFM with ordered probit',
          'ordiu__': 'BFM with ordered probit and implicit user information',
          'ord__ii': 'BFM with ordered probit and implicit movie information',
          'ordiuii': 'BFM with ordered probit, implicit user and movie information'
     }

     fs = os.listdir("./log/")
     for f in fs:
          if f == "log_BFM_iters":
               fs2 = os.listdir("./log/" + f)
               for f2 in fs2:
                    with open("./log/" + f + '/' + f2) as json_file:
                         js = json.load(json_file)
                         val_bfm_iters.append([js["parameters"]["iter"], js["val_rmse"][0]])

          elif f == "log_BFM_options_rank":
               fs2 = os.listdir("./log/" + f)
               for f2 in fs2:
                    with open("./log/" + f + '/' + f2) as json_file:
                         js = json.load(json_file)
                         index = f2[3:10]
                         val_bfm_options_rank[index].append([js["parameters"]["rank"], js["val_rmse"][0]])

          elif f == "log_BFM_options_iters":
               fs2 = os.listdir("./log/" + f)
               for f2 in fs2:
                    with open("./log/" + f + '/' + f2) as json_file:
                         js = json.load(json_file)
                         index = f2[3:10]
                         val_bfm_options_iters[index].append([js["parameters"]["iter"], js["val_rmse"][0]])

          elif f[:3] == "BFM":
               with open("./log/" + f) as json_file:
                    js = json.load(json_file)
                    val_bfm.append([js["parameters"]["rank"], js["val_rmse"][0]])
                    
          elif f[:3] == "SVD":
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

          elif f[:3] == "FSV":
               with open("./log/" + f) as json_file:
                    js = json.load(json_file)
                    val_funk.append([js["parameters"]["rank"], min(js["val_rmse"])])

     
     val_nmf = sorted(val_nmf, key=lambda x: x[0])
     val_svd = sorted(val_svd, key=lambda x: x[0])
     val_als = sorted(val_als, key=lambda x: x[0])
     val_funk = sorted(val_funk, key=lambda x: x[0])
     val_bfm = sorted(val_bfm, key=lambda x: x[0])
     val_bfm_iters = sorted(val_bfm_iters, key=lambda x: x[0])
     val_bfm_options_rank = {key:sorted(value, key=lambda x: x[0]) for (key, value) in val_bfm_options_rank.items()}
     val_bfm_options_iters = {key:sorted(value, key=lambda x: x[0]) for (key, value) in val_bfm_options_iters.items()}
     
     sns.set_style("white")

     # Different options by rank
     for options, val in val_bfm_options_rank.items():
          plt.plot([t[0] for t in val], [t[1] for t in val], '-x', label = options_to_label[options])
     plt.ylim(top=0.995, bottom=0.968)
     plt.ylabel('Validation RMSE', fontsize = 12)
     plt.xlabel('BFM rank', fontsize = 12)
     plt.legend()
     plt.savefig("./plots/BFM_options_by_rank.png")
     plt.cla()

     # Different options by iters
     for options, val in val_bfm_options_iters.items():
          plt.plot([t[0] for t in val], [t[1] for t in val], '-x', label = options_to_label[options])
     plt.ylim(top=0.995, bottom=0.968)
     plt.ylabel('Validation RMSE', fontsize = 12)
     plt.xlabel('BFM iterations', fontsize = 12)
     plt.legend()
     plt.savefig("./plots/BFM_options_by_iters.png")
     plt.cla()

     # Iterations plot for BFM rank 25
     plt.plot([t[0] for t in val_bfm_iters], [t[1] for t in val_bfm_iters], label = 'Validation error')
     plt.ylabel('RMSE', fontsize = 12)
     plt.xlabel('BFM iterations', fontsize = 12)
     plt.legend()
     plt.savefig("./plots/BFM_iters.png")
     plt.cla()
     
     # Ranks plot for BFM with 500 iterations
     plt.plot([t[0] for t in val_bfm], [t[1] for t in val_bfm], label = 'Validation error')
     plt.ylabel('RMSE', fontsize = 12)
     plt.xlabel('BFM rank', fontsize = 12)
     plt.legend()
     plt.savefig("./plots/BFM_ranks.png")
     plt.cla()

     plt.plot([t[0] for t in val_bfm], [t[1] for t in val_bfm], '-x', label = 'BFM')
     plt.plot([t[0] for t in val_svd], [t[1] for t in val_svd], '-x', label = 'SVD', markevery=[5])
     plt.plot([t[0] for t in val_nmf], [t[1] for t in val_nmf], '-x', label = 'NMF', markevery=[9])
     plt.plot([t[0] for t in val_als], [t[1] for t in val_als], '-x', label = 'ALS', markevery=[1])
     plt.plot([t[0] for t in val_funk], [t[1] for t in val_funk], '-x', label = 'FunkSVD', markevery=[0])

     plt.annotate("r=7", (6.4, 1.01))
     plt.annotate("r=11", (10.4, 1.01))
     plt.annotate("r=3", (2.4, 0.9845))
     plt.annotate("r=2", (1.4, 0.997))
     
     plt.xticks(list(range(2,19, 2)) + [22, 26, 30])

     plt.ylim(top = 1.02, bottom=0.968)

     plt.ylabel('Validation RMSE', fontsize = 12)
     plt.xlabel('rank', fontsize = 12)
     plt.legend()
     plt.savefig("./plots/rank_analysis.png")
     plt.cla()



if __name__ == '__main__':
    main()