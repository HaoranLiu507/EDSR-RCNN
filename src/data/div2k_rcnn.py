import os
import cv2
import numpy as np
from data import srdata
from PIL import Image
from src import RCNN
from src.option import args


class DIV2K_RCNN(srdata.SRData):
    """
    This class is used to generate RCNN datasets
    """
    def __init__(self, args, name='DIV2K_RCNN', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        #  If data_range is [['0', '255'], ['0', '255']], after this line of code, self.begin will be 0 and self.end will be 255.
        super(DIV2K_RCNN, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K_RCNN, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K_RCNN, self)._set_filesystem(dir_data)

        # Origin Image Dataset dir
        dir_lr_ori = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        dir_hr_ori = os.path.join(self.apath, 'DIV2K_train_HR')

        # Low-Resolution Image Dataset dir(add RCNN channel)
        dir_lr_rcnn = os.path.join(self.apath, 'DIV2K_RCNN_train_LR_bicubic')

        # Resized Image Dataset dir(add RCNN channel)
        dir_lr_rcnn_resize = os.path.join(self.apath, 'DIV2K_RCNN_resize_train_LR_bicubic')
        dir_hr_rcnn_resize = os.path.join(self.apath, 'DIV2K_RCNN_resize_train_HR')

        # Generate the Corresponding Dataset Folder According to Requirements
        if (not os.path.exists(dir_lr_rcnn_resize) and args.resize == 'on') or (
                not os.path.exists(dir_lr_rcnn) and args.resize == 'off'):

            # Select Batch Processing Image Quantity and Initialize RCNN Model
            batch_size = 200
            model = RCNN.RCNN(args.beta, args.alpha_theta, args.V_theta, args.alpha_U, args.V_U, args.t,
                              args.sigma_kernel,
                              args.sigma_random_closure, args.size, args.rgb_range)

            # Determine Whether to Resize the Dataset
            if args.resize == 'on':
                os.makedirs(dir_lr_rcnn_resize)
                os.makedirs(dir_hr_rcnn_resize)
                target_size_lr = (1100, 1100)
                target_size_hr = tuple(int(args.scale[0])*x for x in target_size_lr)

                # Resize High-Resolution Images in the Specified Folder
                for root, dirs, files in os.walk(dir_hr_ori):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path_hr_resize = os.path.join(root, file)
                            image = Image.open(image_path_hr_resize)
                            image_resized = image.resize(target_size_hr)
                            dest_path = os.path.join(dir_hr_rcnn_resize, os.path.relpath(image_path_hr_resize, dir_hr_ori))
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            image_resized.save(dest_path, format='PNG')
                            image_resized.close()

                # Resize Low-Resolution Images in the Specified Folder
                for root, dirs, files in os.walk(dir_lr_ori):

                    # Process files in batches
                    for i in range(0, len(files), batch_size):
                        batch_files = files[i:i + batch_size]
                        batch_images = []
                        batch_original = []

                        # Load and transform images in the batch
                        for file in batch_files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_path_lr_resize = os.path.join(root, file)
                                image = Image.open(image_path_lr_resize)
                                image_resized = image.resize(target_size_lr)
                                image_np_lr_resized = np.array(image_resized)

                                # Determine whether to convert the image to a grayscale image
                                if args.n_colors == 3:
                                    image_np_ig = cv2.cvtColor(image_np_lr_resized, cv2.COLOR_BGR2GRAY)
                                    batch_images.append(image_np_ig)
                                else:
                                    batch_images.append(image_np_lr_resized)
                                batch_original.append(image_np_lr_resized)
                                image.close()

                        # Convert the batch of processed images to a NumPy array
                        batch_array_lr_resized = np.stack(batch_images, axis=-1)
                        batch_original = np.array(batch_original)

                        if args.n_colors == 3:
                            batch_original = np.transpose(batch_original, (1, 2, 3, 0))
                        else:
                            pass

                        # Apply your RCNN method to process the batch
                        processed_batch = model.RCNN(batch_array_lr_resized, batch_size)
                        processed_batch = np.expand_dims(processed_batch, axis=2)
                        processed_batch = processed_batch.astype(np.uint8)
                        processed_batch_lr = np.concatenate([batch_original, processed_batch], axis=2)

                        # Save the processed batch to the destination path
                        for j, file in enumerate(batch_files):
                            image_path = os.path.join(root, file)
                            RCNN_lr_path_resize = os.path.join(dir_lr_rcnn_resize, os.path.relpath(image_path, dir_lr_ori))
                            os.makedirs(os.path.dirname(RCNN_lr_path_resize), exist_ok=True)
                            if args.n_colors == 3:
                                processed_image_lr = Image.fromarray(processed_batch_lr[:, :, :, j].astype(np.uint8),
                                                                     mode='RGBA')
                            else:
                                processed_image_lr = Image.fromarray(processed_batch_lr[:, :, :, j].astype(np.uint8))
                            processed_image_lr.save(RCNN_lr_path_resize, format='PNG')
                            processed_image_lr.close()

            # No need to resize the image
            else:
                os.makedirs(dir_lr_rcnn)
                for root, dirs, files in os.walk(dir_lr_ori):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path_lr = os.path.join(root, file)
                            image_lr = Image.open(image_path_lr)
                            image_np = np.array(image_lr)
                            if args.n_colors == 3:
                                image_np_ig = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                            else:
                                image_np_ig = image_lr
                            image_lr.close()
                        # Convert the batch of processed images to a NumPy array
                        image_np_ig = np.array(image_np_ig)
                        # Apply your RCNN method to process the batch
                        image_np_ig = np.expand_dims(image_np_ig, axis=-1)
                        processed_batch_lr = model.RCNN(image_np_ig, batch_size=1)
                        if args.n_colors == 3:
                            pass
                        else:
                            image_np = np.expand_dims(image_np, axis=2)
                        processed_batch_lr = processed_batch_lr.astype(np.uint8)
                        processed_batch_lr_final = np.concatenate([image_np, processed_batch_lr], axis=2)
                        # Save the processed batch to the destination path
                        image_path = os.path.join(root, file)
                        RCNN_lr_path = os.path.join(dir_lr_rcnn, os.path.relpath(image_path, dir_lr_ori))
                        os.makedirs(os.path.dirname(RCNN_lr_path), exist_ok=True)
                        if args.n_colors == 3:
                            processed_image = Image.fromarray(processed_batch_lr_final.astype(np.uint8),
                                                              mode='RGBA')
                        else:
                            processed_image = Image.fromarray(processed_batch_lr_final.astype(np.uint8))
                        processed_image.save(RCNN_lr_path)
                        processed_image.close()

        # Determine the dataset path based on whether resizing is needed.
        if args.resize == 'on':
            self.dir_lr = dir_lr_rcnn_resize
            self.dir_hr = dir_hr_rcnn_resize
        else:
            self.dir_lr = dir_lr_rcnn
            self.dir_hr = dir_hr_ori

        if self.input_large: self.dir_lr += 'L'
