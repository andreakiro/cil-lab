import wandb
import os

def activate_wnb(args):
    wandb_config = dict()

    if args.wandb == 'offline':
        os.environ['WANDB_SILENT'] = 'true'

    if args.model == 'deeprec':
        wandb_config = {
            # model architecture
            'architecture': 'deeprec',
            'major': args.major,
            'activation': args.activation,
            'layer1_dim': args.layer1_dim,
            'layer2_dim': args.layer2_dim,
            'layer3_dim': args.layer3_dim,
            # training hyperparams
            'epochs': args.epochs,
            'optimizer': args.optimizer,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            # regularization params
            'weight_decay': args.weight_decay,
            'dropout': args.dropout,
            'noise_prob': args.noise_prob,
            'dense_refeeding_steps': args.dense_refeeding_steps,
        }

    if args.model == 'lightgcn':
        wandb_config= {
            # model architecture
            'architecture': 'lightgcn',
            'num_layers': args.num_layers,
            'emb_size': args.emb_size,
            # training hyperparams
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
        }

    wandb.init(
        project='cil-lab',
        config = wandb_config,
        mode = args.wandb,
        job_type = args.mode,
        resume = 'auto',
    )

    # enable wandb run names
    # args.rname = wandb.run.name