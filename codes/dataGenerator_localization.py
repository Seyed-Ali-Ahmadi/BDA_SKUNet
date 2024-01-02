class xBD_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path_to_jsons, batch_size=5, patch_size=256, shuffle=True, preprocessing=None):

        self.path_to_jsons = path_to_jsons
        self.pre_img_paths = []
        self.loc_target_paths = []

        for i in range(len(self.path_to_jsons)):
            self.pre_img_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(self.path_to_jsons[i].rsplit('/', 3)[1:]).replace('/labels/', '/images/').replace('_post_', '_pre_') + '.png')
            if 'tier3' not in self.path_to_jsons[i]:
              self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(self.path_to_jsons[i].rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '_target.png')
            else:
              self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(self.path_to_jsons[i].rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '.png')

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.on_epoch_end()
        self.preprocessing = preprocessing

    def __len__(self):
        return int(len(self.pre_img_paths) // self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        pre_image_batch = list(np.array(self.pre_img_paths)[batch_indices])
        loc_target_batch = list(np.array(self.loc_target_paths)[batch_indices])

        x_pre_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype='float32')
        y_target_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (1,), dtype='float32')
        for b in range(self.batch_size):
            pre_image = imread(pre_image_batch[b])
            target_image = imread(loc_target_batch[b])

            # AUGMENTATION TO BE ADDED #
            transform = A.Compose([
                A.RandomCrop(width=self.patch_size, height=self.patch_size, p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(p=0.6),
                A.Transpose(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(min_holes=4, max_holes=12, max_height=20, max_width=20, mask_fill_value=0, p=0.6),     # https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/
                A.OneOf([
                    A.MotionBlur(p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.6),
                    A.Blur(blur_limit=3, p=0.6)], p=0.7),])
            transformed = transform(image=pre_image, mask=target_image)

            if self.preprocessing is None:
                x_pre_batch[b] = transformed['image']
            else:
                x_pre_batch[b] = self.preprocessing(transformed['image'])
            y_target_batch[b, :, :, 0] = transformed['mask']

        return x_pre_batch, y_target_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
