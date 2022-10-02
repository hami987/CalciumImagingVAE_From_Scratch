import vae
from Dataloader import setup_data_loaders
import os
import datetime
import argparse

from_ckpt = True
ckpt_path = "checkpoint_000.tar"
img_range = [0,100]

def str2bool(v):
    """
    Str to Bool converter for wrapper script.
    This is used both for --from_ckpt flag, which
    is False by default but can be turned on either by listing the flag (without args)
    or by listing with an appropriate arg (which can be converted to a corresponding boolean)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="User data for VAE")

    parser.add_argument('--data_dir', type=str, metavar='N', default='./Data/', required=False, \
                        help='Path to directory where data lives.')
    parser.add_argument('--save_dir', type=str, metavar='N', default=f'./Saves/{datetime.datetime.now().date().strftime("%d_%m_%Y")}', \
                        help='Dir where model params, latent projection maps and TB logs are saved to. Default is to save files to current dir.')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', \
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N', \
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', \
                        help='Random seed (default: 1)')
    parser.add_argument('--save_freq', type=int, default=2, metavar='N', \
                        help='How many epochs to wait before saving training status.')
    parser.add_argument('--test_freq', type=int, default=1, metavar='N',
                        help='How many epochs to wait before testing.')
    parser.add_argument('--from_ckpt', type=str2bool, nargs='?', const=True, default=False,
                        help='Boolean flag indicating if training and/or reconstruction should be carried using a pre-trained model state.')
    parser.add_argument('--ckpt_path', type=str, metavar='N', default='',
                        help='Path to ckpt with saved model state to be loaded. Only effective if --from_ckpt == True.')

    args = parser.parse_args()

    loaders = setup_data_loaders(data_dir=args.data_dir)
    model = vae.BehaviourVAE()
    if from_ckpt == args.from_ckpt:
        print(os.getcwd())
        assert os.path.exists(args.ckpt_path), 'Oops, looks like ckpt file given does NOT exist!'
        print('='*40)
        print('Loading model state from: {}'.format(args.ckpt_path))
        model.load_state(filename = args.ckpt_path)
    #model.train_loop(loaders,epochs=args.epochs)

    # recnstructions
    #recons = model.get_recons(loaders['dset'], [img_range[0], img_range[1]])
    #print(f"All Recons Shape: {recons.shape}")
    #All Recons Shape: (100, 6, 608, 608)

    projections = model.get_latent_umap(loaders, save_dir="", title="Latent Space plot")
