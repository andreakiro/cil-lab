from matplotlib import pyplot as plt
import json
import os


def main():

     train = []
     val = []

     fs = os.listdir("./log/")
     for f in fs:
          if f[:3] == "SVD":
               with open("./log/" + f) as json_file:
                    js = json.load(json_file)
                    train.append([js["parameters"]["rank"], js["train_rmse"][0]])
                    val.append([js["parameters"]["rank"], js["val_rmse"][0]])
     
     train = sorted(train, key=lambda x: x[0])
     val = sorted(val, key=lambda x: x[0])

     plt.style.use('seaborn')
     #plt.plot([t[0] for t in train], [t[1] for t in train], label = 'Training error')
     plt.plot([t[0] for t in val], [t[1] for t in val], label = 'Validation error')
     plt.ylabel('RMSE', fontsize = 14)
     plt.xlabel('SVD rank', fontsize = 14)
     plt.title('Learning curves.', fontsize = 18, y = 1.03)
     plt.legend()
     plt.savefig("./plots/SVD.png")


if __name__ == '__main__':
    main()