import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import os
import datetime
import itertools
import torch
from torch.distributions import LowRankMultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from umap import UMAP

X_SHAPE = (6, 1, 608, 608) #dimensions of the inidividual videos
X_DIM = np.prod(X_SHAPE)

class BehaviourVAE(nn.Module):
    def __init__(self, lr=1e-4, z_dim=32,device_name="auto",model_precision=10,save_dir=""):
        super(BehaviourVAE, self).__init__()
        self.save_dir = save_dir
        self.lr = lr
        self.z_dim = z_dim
        self.model_precision = model_precision

        assert device_name != "cuda" or torch.cuda.is_available()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        print(f"Using device: {device_name}")

        if self.save_dir != "" and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._build_network()
        self.optimizer = Adam(self.parameters(),lr=self.lr)
        self.epoch = 0
        self.loss = {"train":{}, "test":{}}

        ts = datetime.datetime.now().date()
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir,"run",ts.strftime("%m_%d_%Y")))
        #self.to(self.device)

    def _build_network(self):
        # Encoder
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(1)
        self.bn2 = nn.BatchNorm3d(8)
        self.bn3 = nn.BatchNorm3d(16)
        self.bn4 = nn.BatchNorm3d(16)

        self.fc1 = nn.Linear(46208, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc31 = nn.Linear(256, 64)
        self.fc32 = nn.Linear(256, 64)
        self.fc33 = nn.Linear(256, 64)
        self.fc41 = nn.Linear(64, self.z_dim)
        self.fc42 = nn.Linear(64, self.z_dim)
        self.fc43 = nn.Linear(64, self.z_dim)

        # Decoder
        self.fc5 = nn.Linear(self.z_dim, 64)
        self.fc6 = nn.Linear(64, 256)
        self.fc7 = nn.Linear(256, 1024)
        self.fc8 = nn.Linear(1024, 46208)

        # torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.convt1 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=(1, 1, 1))
        self.convt2 = nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=(1, 1, 1),
                                         output_padding=(1, 1, 0))
        self.convt3 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=0)
        self.convt4 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=1, padding=0)

        self.bn5 = nn.BatchNorm3d(32)
        self.bn6 = nn.BatchNorm3d(16)
        self.bn7 = nn.BatchNorm3d(16)
        self.bn8 = nn.BatchNorm3d(8)

    def _get_layers(self):
        """Return a dictionary mapping names to network layers."""
        return {'fc1': self.fc1, 'fc2': self.fc2, 'fc31': self.fc31,
                'fc32': self.fc32, 'fc33': self.fc33, 'fc41': self.fc41,
                'fc42': self.fc42, 'fc43': self.fc43, 'fc5': self.fc5,
                'fc6': self.fc6, 'fc7': self.fc7, 'fc8': self.fc8, 'bn1': self.bn1,
                'bn2': self.bn2, 'bn3': self.bn3, 'bn4': self.bn4, 'bn5': self.bn5,
                'bn6': self.bn6, 'bn7': self.bn7, 'bn8': self.bn8, 'conv1': self.conv1,
                'conv2': self.conv2, 'conv3': self.conv3, 'conv4': self.conv4,
                'convt1': self.convt1, 'convt2': self.convt2,
                'convt3': self.convt3, 'convt4': self.convt4}

    def encode(self, x):

        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = x.view(-1, 46208)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.relu(self.fc31(x))
        mu = self.fc41(mu)
        u = F.relu(self.fc32(x))
        u = self.fc42(u).unsqueeze(-1)  # Last dimension is rank \Sigma = 1.
        d = F.relu(self.fc33(x))
        d = torch.exp(self.fc43(d))  # d must be positive.
        return mu, u, d

    def decode(self, z):
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = z.view(-1, 32, 152, 152, 2)
        z = F.relu(self.convt1(self.bn5(z)))
        z = F.relu(self.convt2(self.bn6(z)))
        z = F.relu(self.convt3(self.bn7(z)))
        z = F.relu(self.convt4(self.bn8(z)))
        z = z.view(-1, X_DIM)
        return z

    def forward(self,x,return_latent_rec=False,train_mode=False):

        mu, u, d = self.encode(x)
        latent_dist = LowRankMultivariateNormal(mu, u, d)
        z = latent_dist.rsample()
        x_rec = self.decode(z)
        if train_mode:  # log reconstructions
            batch_size = x_rec.detach().cpu().numpy().shape[0]
            self.log_reconstruction(x_rec.detach().cpu().numpy(), batch_size, log_type='train')
        # E_{q(z|x)} p(z)
        elbo = -0.5 * (torch.sum(torch.pow(z, 2)) + self.z_dim * np.log(2 * np.pi))
        # E_{q(z|x)} p(x|z)
        pxz_term = -0.5 * X_DIM * (np.log(2 * np.pi / self.model_precision))
        #l2s = torch.sum(torch.pow(x.view(x.shape[0], -1) - x_rec, 2), dim=1)
        l2s = torch.sum(torch.pow(x.reshape(x.shape[0], -1) - x_rec, 2), dim=1)
        pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s)
        elbo = elbo + pxz_term
        # H[q(z|x)]
        elbo = elbo + torch.sum(latent_dist.entropy())
        if return_latent_rec:
            return -elbo, z.detach().cpu().numpy(), \
                   x_rec.view(-1, X_SHAPE[0], X_SHAPE[1], X_SHAPE[2], X_SHAPE[3]).detach().cpu().numpy()
        return -elbo

    def train_epoch(self,train_loader):
        self.train()
        train_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            print(f"Batch_idx: {batch_idx}")
            self.optimizer.zero_grad()
            frame = data['frame']
            #frame = frame.to(self.device)
            loss, z, x_rec = self.forward(frame, train_mode=True, return_latent_rec=True)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(train_loader.dataset)
        print(f"Z Shape: {z.shape}")
        print(f"x_rec Shape: {x_rec.shape}")
        print('Epoch: {} Average loss: {:.4f}'.format(self.epoch,train_loss))
        self.epoch += 1
        return train_loss

    def test_epoch(self,test_loader):
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                frame = data['frame']
                #frame = frame.to(self.device)
                loss = self.forward(frame)
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
        # log loss to TB
        self.writer.add_scalar("Loss/Test", test_loss, self.epoch)
        self.writer.flush()
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss

    def train_loop(self, loaders, epochs=4, test_freq=1, save_freq=1):
        print("="*40)
        print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
        print("Training set:", len(loaders['train'].dataset))
        print("Test set:", len(loaders['test'].dataset))
        print("="*40)
        # For some number of epochs...
        for epoch in range(self.epoch, self.epoch+epochs):
            # Run through the training data and record a loss.
            print(f"Epoch {epoch} / {epochs}")
            loss = self.train_epoch(loaders['train'])
            self.loss['train'][epoch] = loss
            #log loss to TB
            self.writer.add_scalar("Loss/Train", loss, self.epoch)
            self.writer.flush()
            # Run through the test data and record a loss.
            if (test_freq is not None) and (epoch % test_freq == 0):
                loss = self.test_epoch(loaders['test'])
                self.loss['test'][epoch] = loss
            # Save the model.
            if (save_freq is not None) and (epoch % save_freq == 0) and (epoch > 0):
                filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
                self.save_state(filename)

    def save_state(self, filename):
        """Save all the model parameters to the given file."""
        layers = self._get_layers()
        state = {}
        for layer_name in layers:
            state[layer_name] = layers[layer_name].state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()
        state['loss'] = self.loss
        state['z_dim'] = self.z_dim
        state['epoch'] = self.epoch
        state['lr'] = self.lr
        state['save_dir'] = self.save_dir
        filename = os.path.join(self.save_dir, filename)
        torch.save(state, filename)
        print("Checkpoint saved")

    def load_state(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        assert checkpoint['z_dim'] == self.z_dim
        layers = self._get_layers()
        for layer_name in layers:
            layer = layers[layer_name]
            layer.load_state_dict(checkpoint[layer_name])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']

    def log_reconstruction(self, frames, batch_size, log_type):
        frames = frames.reshape((batch_size, X_SHAPE[0], X_SHAPE[1], X_SHAPE[2], X_SHAPE[3]))

        for i in range(batch_size):
            fig_name = 'reconstruction_{}/{}'.format(log_type, i)
            frame = frames[i, :, :, :, :]
            """print(f"Frame Shape before permute: {frame.shape}")
            #frame = frame.resize(1,2,3,0)
            print(f"Frame Shape in Plotting: {frame.shape}")

            plt.imshow(frame[0,0,:,:], cmap='gray', vmin=0, vmax=1)
            plt.show()"""
            self.writer.add_image(fig_name, frame[0, :, :, :], dataformats="CHW")

    def get_recons(self, dataset, vals_list):
        """
        Returns a np array with recons for all frames in train set (in order).
        Useful when wanting to build tooltip plot.

        Args:
        ---------
        Dataset: dataclass instance.
        Vals_list: (list). First element is starting frame (inclusive) and
        last is ending frame (exclusive).
        """
        start = vals_list[0]
        end = vals_list[1]
        all_recons = []
        for i in range(start, end):
            frame = dataset.__getitem__(i)
            frame = frame['frame'].unsqueeze(0)
            #print(f"Frame Shape in get_recons: {frame.shape}")
            #frame = frame.to(self.device)
            _, _, recon = self.forward(frame, return_latent_rec=True)
            #print(f"Recon Shape in get_recos: {recon.shape}")
            all_recons.append(np.squeeze(recon))
        return np.array(all_recons)

    def get_latent_umap(self, loaders, save_dir, title=None):
        """UMAP
        Not sure if these are mapped correctly or if there are still dimension mismatches here."""

        filename = str(self.epoch).zfill(3) + '_latents.pdf'
        file_path = os.path.join(save_dir, filename)
        print(f"Test-Set Length: {len(loaders['test'].dataset)}")
        print(f"z-dim: {self.z_dim}")
        latent = np.zeros((len(loaders['test'].dataset), self.z_dim)) #am using test loader b/c it is unshuffled.
        #latent = np.zeros((384, self.z_dim))
        print(f"Latent-0-Matrix: {latent.shape}")
        with torch.no_grad():
            j = 0
            for i, sample in enumerate(loaders['test']):
                x = sample['frame']
                #x = x.to(self.device)
                mu, _, _ = self.encode(x)
                print(f"Mu Shape: {mu.shape}")
                print(f"Mu length: {len(mu)}")
                print(f"j: {j}")
                #print(f"Latent Shape: {latent.shape}")
                latent[j:j+len(mu)] = mu.detach().cpu().numpy()
                j += len(mu)
        print(f"Latent-Matrix: {latent.shape}")
        # UMAP these
        transform = UMAP(n_components=2, n_neighbors=20, min_dist=0.1,metric='euclidean', random_state=42)
        projection = transform.fit_transform(latent)
        # save these to do PCA and the rest
        latent_info = {'latents':latent, 'UMAP':projection}
        fname = os.path.join(self.save_dir, 'latent_info.tar')
        torch.save(latent_info, fname)
        #and return projections for plotting
        return projection