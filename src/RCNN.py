import torch
import torch.cuda
import torch.nn.functional as F
import cv2
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

device = torch.device("cuda")
import numpy as np


class RCNN:

    def __init__(self, beta , alpha_theta, V_theta, alpha_U, V_U, t, sigma_kernel, sigma_random_closure, size, rgb_range):
        """
        This Class performs RCNN-based feature extraction for multiple grayscale images (given in a numpy
        ndarray), where the number of images depends on the given batch size. When the process is down, this funtion
        outputs ignition maps that have the same size as the input numpy ndarray.

        :param images: High-dimensional matrix obtained by merging multiple grayscale images
        :param batch_size: Batch processing size for feature extraction
        :param beta: Weighting factor that controls the relationship between feedback and link inputs
        :param alpha_theta: Dynamic threshold decay coefficient
        :param V_theta: Dynamic threshold weighting coefficient
        :param alpha_U: Internal activity decay coefficient
        :param V_U: Internal activity weighting coefficient
        :param t: Number of iterations for RCNN ignition
        :param sigma_kernel:Variance of 2-D Gaussian distribution for Gaussian kernel matrix
        :param sigma_random_closure:Variance of 2-D Gaussian distribution for random closure probability matrix
        :param size: Gaussian kernel size (size by size)
        :param rgb_range: RGB range of the image/video (eg, 255 for 8 bit images, and 65536 for 16 bit images)
        """
        # Initialize parameters
        self.beta = beta
        self.alpha_theta = torch.tensor(alpha_theta)
        self.V_theta = V_theta
        self.alpha_U = torch.tensor(alpha_U)
        self.V_U = V_U
        self.t = t
        self.sigma_kernel = sigma_kernel
        self.sigma_random_closure = sigma_random_closure
        self.size = size
        self.rgb_range = rgb_range

    def RCNN(self, images, batch_size=1):
        # Cook the input images in preparation for latter processing
        images = torch.from_numpy(self.images_norm(images)).to(device)

        # Declare the variables and move them to the device
        [m, n, c] = images.shape
        ignition_map = torch.zeros([m, n, c]).to(device)
        U = ignition_map
        threshold = ignition_map + 1
        neuron_output = ignition_map.double().to(device)
        self.gaussian_kernel_matrix = self.get_gaussian_kernel(dimension=self.size, sigma=self.sigma_kernel)
        self.gaussian_kernel_matrix[int((self.size + 1) / 2), int((self.size + 1) / 2)] = 0
        self.gaussian_kernel_matrix = torch.unsqueeze(self.gaussian_kernel_matrix, dim=0)
        self.random_closure_probability_matrix = self.get_gaussian_kernel(dimension=self.size, sigma=self.sigma_random_closure)
        weight_default = self.gaussian_kernel_matrix.repeat(batch_size, 1, 1, 1)

        # Ignition iterations
        for i in range(self.t):
            # Generate the random closure matrix
            mask = self.random_closure(self.size, 0.1, 'Gaussian', batch_size)

            # Random closure
            weight = torch.where(mask, weight_default, torch.zeros_like(weight_default))

            # Link input
            L = F.conv2d(input=neuron_output.reshape([1, batch_size, m, n]), weight=weight, bias=None, stride=1,
                         padding=self.size // 2, dilation=1, groups=int(batch_size)).squeeze().reshape([m, n, c])

            # Neural internal activity
            U = torch.exp(-self.alpha_U) * U + images * (1 + self.beta * self.V_U * L)

            # Neuron ignition
            neuron_output = (U > threshold).double()

            # Update dynamic threshold
            threshold = torch.mul(torch.exp(-self.alpha_theta), threshold) + self.V_theta * neuron_output

            # Sum ignition results
            ignition_map = ignition_map + neuron_output

        return ignition_map.cpu().numpy()

    @staticmethod
    def get_gaussian_kernel(dimension, sigma):
        """
        Generate two-dimensional Gaussian kernel.

        :param dimension: Gaussian kernel size
        :param sigma: Variance of 2-D Gaussian distribution
        """
        kernel = torch.from_numpy(cv2.getGaussianKernel(dimension, sigma)).to(device)
        transpose_kernel = torch.from_numpy(cv2.getGaussianKernel(dimension, sigma).T).to(device)
        matrix = torch.multiply(kernel, transpose_kernel)

        return matrix

    def images_norm(self, images):
        """
        Convert the pixel values in the image to the data type of float, and normalize it into range 0-1.

        :param images: high-dimensional matrix obtained by merging multiple grayscale images
        """
        return images.astype(np.float32) / self.rgb_range

    def random_closure(self, dimension, P, flag, batch_size):
        """
        Generate a random closure matrix to modulate the weight contribution of neurons. It is composed of 0 and 1,
        where 1 represents that the connection input between the central nerve and the neuron at that location is
        turned on, while 0 represents that the connection input is turned off, also known as neural connection
        random_closure.

        :param dimension: Size of weight matrix
        :param P: random closure probability for uniform distribution
        :param flag: random closure type (Optional: "Gaussian" or "uniform")
                     when assigned to "Gaussian", the random closure probability follows two-dimensional Gaussian distribution
                     (i.e. the random closure probability is proportional to the distance from the central neuron);
                     when assigned to "uniform", the random closure probability follows uniform distribution between 0 and 1
                     (i.e. the random closure probability is the same across whole kernel)
        :param batch_size: batch processing size
        """

        if flag == 'Gaussian':
            # Cook the random closure probability matrix to meet the batch processing requirements
            random_closure_probability = self.random_closure_probability_matrix.unsqueeze(0).unsqueeze(0).repeat(
                batch_size, 1, 1, 1)

            # Normalize the probability into range 0-1
            random_closure_probability = random_closure_probability / random_closure_probability[
                0, 0, int((dimension + 1) / 2),
                int((dimension + 1) / 2)]

            # Generate random number between 0-1
            random_number = torch.rand(batch_size, 1, dimension, dimension).to(device)

            # random closure
            matrix = random_number < random_closure_probability

        if flag == 'uniform':
            # Generate random number between 0-1
            random_number = torch.rand(batch_size, 1, dimension, dimension).to(device)

            # Constant random closure probability
            random_closure_probability = torch.ones(batch_size, 1, dimension, dimension, device=device) * P

            # random closure
            matrix = random_number < random_closure_probability

        return matrix
