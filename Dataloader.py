import random
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as torch_tf
import torchvision.transforms.functional as torch_func
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

class VideoDataLoader(Dataset):
    def __init__(self, video_path, transforms:list=[]):

        videos = random.sample(os.listdir(os.path.join(video_path)),3)
        self.all_frames = []

        for video in videos:

            print(video)

            frames,_,_ = torchvision.io.read_video(os.path.join(video_path,str(video)))
            self.frames = frames.permute(0,3,2,1).double()
            if len(self.frames) > 100:
                self.frames = self.frames[:100]

            if transforms is not None:
                self.frames = torch_tf.functional.rgb_to_grayscale(self.frames)
                self.frames = torch_tf.functional.normalize(self.frames,mean=[0],std=[255])
            self.frames = self.frames.permute(1,2,3,0)
            np_arr = self.frames.cpu().detach().numpy()
            self.all_frames.append(self.frames)
            """self.df = np_arr
            self.max = np.amax(self.df.flatten())
            self.min = np.amin(self.df.flatten())
            self.mean = np.mean(self.df.flatten())
            print(self.max)
            print(self.min)"""

        self.all_frames = torch.cat([tensor for tensor in self.all_frames], dim=-1)
        np_arr = self.all_frames.cpu().detach().numpy()
        self.df = np_arr
        # dimensions: (1, 640, 480, 300)
        # dimensions: (channels, height, width, frames)
        self.max = np.amax(self.df.flatten())
        self.min = np.amin(self.df.flatten())
        self.mean = np.mean(self.df.flatten())
        print(self.max)
        print(self.min)



    def transform_to_gray_scale(self):

        frames = torch_tf.functional.rgb_to_grayscale(self.frames)
        return frames

    def __len__(self):
        """
        Returns number of samples in dset
        """
        return (int(self.df.shape[3] - 6))

    def __getitem__(self, idx):
        """
        Returns a single sample from dset.
        """
        time_start = idx
        time_end = time_start + 6#+ 10 #am picking 31 frames at time here, this might be too much for your data!
        frame = self.df[:, :, :,time_start:time_end]
        scld_frame = np.true_divide((frame - self.min), (self.max - self.min)) #min/max norm (global)
        sample = {'frame': torch.from_numpy(scld_frame).float()}
        return sample

    def plot(self):
        print(f"Frame Shape in Plotting: {self.frames[-1].shape}")

        plt.imshow(self.frames[-1][:,:,50], cmap='gray', vmin=0, vmax=1)
        plt.show()


def setup_data_loaders(batch_size=6, shuffle=(True,False), data_dir='./Data/'):
    dset = VideoDataLoader(data_dir, transforms=["grey"])
    #dset.plot()
    train_loader = DataLoader(dset, batch_size=batch_size,shuffle=shuffle[0],num_workers=0)
    test_loader = DataLoader(dset,batch_size=batch_size,shuffle=shuffle[1],num_workers=0)
    return {'train': train_loader, 'test': test_loader, 'dset': dset}




if __name__ == '__main__':
    loaders = setup_data_loaders()
    for batch_idx, data in enumerate(loaders["train"]):
        print(f"Batch_idx: {batch_idx}")
        print(f"Data Shape: {data['frame'].shape}")