import glob
import os.path
from torch.utils.data import Dataset
import nibabel as nib
from multiprocessing import Manager
import torchio as tio
import pickle
import torch
import numpy as np
def map_image(img, out_affine, out_shape, ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
              order=1):
    """
    Function to map image to new voxel space (RAS orientation)
    :param nibabel.MGHImage img: the src 3D image with data and affine set
    :param np.ndarray out_affine: trg image affine
    :param np.ndarray out_shape: the trg shape information
    :param np.ndarray ras2ras: ras2ras an additional maping that should be applied (default=id to just reslice)
    :param int order: order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: np.ndarray new_data: mapped image data array
    """
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    # compute vox2vox from src to trg
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asanyarray(img.dataobj)
    # convert frames to single image
    if len(image_data.shape) > 3:
        if any(s != 1 for s in image_data.shape[3:]):
            raise ValueError(f'Multiple input frames {tuple(image_data.shape)} not supported!')
        image_data = np.squeeze(image_data, axis=tuple(range(3,len(image_data.shape))))

    new_data = affine_transform(image_data, inv(vox2vox), output_shape=out_shape, order=order)
    return new_data

def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.
    :param np.ndarray data: image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: float src_min: (adjusted) offset
    :return: float scale: scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale


def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    Function to crop the intensity ranges to specific min and max values
    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: np.ndarray data_new: scaled image data
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new

def conform(img, order=1):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.
    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again
    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader

    cwidth = 256
    csize = 1
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    src_min, scale = getscale(np.asanyarray(img.dataobj), 0, 255)

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))

    if not img.get_data_dtype() == np.dtype(np.uint8):

        mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img
def spatial_conform(img, order=1):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.
    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again
    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader

    cwidth = 256
    csize = 1
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords


    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    return mapped_data

def Generate_dataset():
    # This if else sentence is used to decide whether enables dataset cache funciton
    if False:
        cache = DatasetCache(None,use_cache=False)
        manager2 = Manager()
        cache2 = DatasetCache(manager2,use_cache=True)
    else:
        manager = None
        cache = DatasetCache(manager,use_cache=False)
        manager2 = None
        cache2 = DatasetCache(manager2,use_cache=False)


    train_dataset = VoxelDataset(cache=cache, train=True)
    val_dataset = VoxelDataset( cache=cache2, train=False)

    return train_dataset,val_dataset
'''
import matplotlib.pyplot as plt
plt.imshow(nib.load(self.intensity_spatial_norm_file_path[5894]).get_fdata()[64], cmap='gray')
plt.show()
'''
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        if self.manager is not None:
            self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, img, lbl):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (img, lbl)


class VoxelDataset(Dataset):
    def __init__(self, rescale_sdf=True, cache = None,train=False):
        # note that input image paths are for processed images rather than unprocessed
        skull_stripping_dataset_path = '/home/xh278/skull_stripped_dataset/NFBS_Dataset/*/*-NFB3_T1w.nii.gz'

        self.skull_stripping_dataset_orig_file_path = glob.glob(skull_stripping_dataset_path)
        self.skull_stripping_dataset_seg_file_path = [i.replace('NFB3_T1w.nii.gz','NFB3_T1w_brainmask.nii.gz') for i in self.skull_stripping_dataset_orig_file_path]
        # For PPMI dataset, the corresponding folder contains average folder, we need to root them out

        self.orig_file_path = self.skull_stripping_dataset_orig_file_path

        self.rescale_sdf = rescale_sdf
        self.cache = cache
        self.train = train

    def __len__(self):
        return len(self.orig_file_path)

    def __getitem__(self, index):
        image_resolution= 128
        normalization=255
        output_index = [index]
        for cur_index in output_index:
            img_path = self.orig_file_path[cur_index]
            array = nib.load(img_path)
            if 'nii.gz' in img_path or 'nii' in img_path:
                array = conform(array)
            input = array.get_fdata()
            input = torch.FloatTensor(input)
            input = input.unsqueeze(0).unsqueeze(0)
            if image_resolution != 256:
                input = torch.nn.functional.interpolate(input, size=[image_resolution, image_resolution, image_resolution], mode='trilinear',align_corners=False)[0]
            else:
                input = torch.Tensor(input)[0]

            input = input.clip(0, normalization) / normalization

            seg = nib.load(self.skull_stripping_dataset_seg_file_path[cur_index])

            seg = spatial_conform(seg)
            seg = seg>0
            seg = torch.FloatTensor(seg).unsqueeze(0).unsqueeze(0)
            seg = torch.nn.functional.interpolate(seg, size=[image_resolution, image_resolution, image_resolution],
                                                mode='nearest')[0]

        return input,seg,img_path



