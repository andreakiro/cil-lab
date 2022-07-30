# Define parser with full params
# Default currently set at optimal
##################################

from src.configs import config

def fill_deeprec_ps(deeprec_ps):
    # select model, mode and run name
    deeprec_ps.add_argument('--mode', type=str, help='define mode for experience', choices={'train', 'test'}, required=True)
    deeprec_ps.add_argument('--rname', type=str, help='name of the experiment when wandb is offline', required=True)
    deeprec_ps.add_argument('--path_to_model', type=str, help='restoring model e.g. when testing')

    # paths to data and logs
    deeprec_ps.add_argument('--path_to_train_data', type=str, default='data/training.data', help='path to training data')
    deeprec_ps.add_argument('--path_to_eval_data', type=str, default='data/validation.data', help='path to evaluation data')
    deeprec_ps.add_argument('--path_to_test_data', type=str, default='data/testing.data', help='path to testing data')
    deeprec_ps.add_argument('--logdir', type=str, default='logs', help='where to save model and write logs')

    # model architecture
    deeprec_ps.add_argument('--major', type=str, default='users', help='major of the model (users or items)')
    deeprec_ps.add_argument('--activation', type=str, default='selu', help='type of the non-linearity used in activations')
    deeprec_ps.add_argument('--layer1_dim', type=int, default=512, help='dimension of the hidden layer 1')
    deeprec_ps.add_argument('--layer2_dim', type=int, default=128, help='dimension of the hidden layer 2')
    deeprec_ps.add_argument('--layer3_dim', type=int, default=1024, help='dimension of the hidden layer 3')

    # training parameters
    deeprec_ps.add_argument('--epochs', type=int, default=300, help='maximum number of epochs for training')
    deeprec_ps.add_argument('--optimizer', type=str, default='momentum', help='optimizer kind: adam, momentum, adagrad or rmsprop')
    deeprec_ps.add_argument('--learning_rate', type=float, default=0.01, help='global learning rate for training')
    deeprec_ps.add_argument('--batch_size', type=int, default=128, help='global batch size for training')

    # regularization parameters
    deeprec_ps.add_argument('--weight_decay', type=float, default=0.001, help='L2 weight decay factor')
    deeprec_ps.add_argument('--dropout', type=float, default=0.2, help='dropout probabilitz during training')
    deeprec_ps.add_argument('--noise_prob', type=float, default=0.0, help='decoding noise probability')
    deeprec_ps.add_argument('--dense_refeeding_steps', type=int, default=2, help='freq. of data augmentation')

    # other parameters
    deeprec_ps.add_argument('--constrained', type=bool, default=True, help='constraints definition on autoencoder')
    deeprec_ps.add_argument('--skip_last_layer_nl', type=bool, default=True, help='application of non-linearity on last layer')
    deeprec_ps.add_argument('--num_checkpoints', type=int, default=4, help='number of saved model checkpoints (including last)')
    deeprec_ps.add_argument('--evaluation_frequency', type=int, default=1, help='frequency (epoch-based) of model evaluation')
    deeprec_ps.add_argument('--summary_frequency', type=int, default=100, metavar='N', help='how often to save summaries')
    deeprec_ps.add_argument('--hidden_layers', type=str, default='512,128,1024', metavar='N', help='hidden layer sizes, comma-separated')
    deeprec_ps.add_argument('--gpu_ids', type=str, default='0', help='comma-separated gpu ids to use for data parallel training')

    return deeprec_ps

def fill_lightgcn_ps(lightgcn_ps):
    # select model, mode and run name
    lightgcn_ps.add_argument('--mode', type=str, help='define mode for experience', choices={'train', 'test'}, required=True)
    lightgcn_ps.add_argument('--rname', type=str, help='name of the experiment when wandb is offline', required=True)
    lightgcn_ps.add_argument('--path_to_model', type=str, help='restoring model e.g. when testing')

    # training parameters
    lightgcn_ps.add_argument('--epochs', type=int, default=1, help='maximum number of epochs for training')
    lightgcn_ps.add_argument('--learning_rate', type=float, default=0.0001, help='global learning rate for training')
    lightgcn_ps.add_argument('--batch_size', type=int, default=2048, help='global batch size for training')

    # model architecture for lightgcn
    lightgcn_ps.add_argument('--num_layers', type=int, default=4, help='num of layers i.e. iterations of agg function in model')
    lightgcn_ps.add_argument('--emb_size', type=int, default=64, help='layer embedding size of the model')

    # paths to data and logs
    lightgcn_ps.add_argument('--r_train_path', type=str, default=config.lightgcn.R_MATRIX_TRAIN, help='path to R matrix needed for training')
    lightgcn_ps.add_argument('--r_mask_train_path', type=str, default=config.lightgcn.R_MASK_TRAIN, help='path to R mask matrix needed for training')

    # other parameters
    lightgcn_ps.add_argument('--print_freq', type=int, default=50, help='frequency of training loss prints (batch-based)')
    lightgcn_ps.add_argument('--eval_freq', type=int, default=1, help='frequency of model evaluation (epoch-based)')

    return lightgcn_ps