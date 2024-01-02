class xBD_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path_to_jsons, batch_size=5, patch_size=256, shuffle=True, preprocessing=None):

        self.path_to_jsons = path_to_jsons
        self.pre_img_paths = []
        self.post_img_paths = []
        self.loc_target_paths = []
        self.cls_target_paths = []

        for path in self.path_to_jsons:
            self.pre_img_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/images/').replace('_post_', '_pre_') + '.png')
            self.post_img_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/images/') + '.png')
            if 'tier3' not in path:
                self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '_target.png')
                self.cls_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/') + '_target.png')
            else:
                self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '.png')
                self.cls_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/') + '.png')

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
        post_image_batch = list(np.array(self.post_img_paths)[batch_indices])
        loc_target_batch = list(np.array(self.loc_target_paths)[batch_indices])
        cls_target_batch = list(np.array(self.cls_target_paths)[batch_indices])

        x_pre_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype='float32')
        x_post_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype='float32')
        y_target_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (5,), dtype='float32')
        for b in range(self.batch_size):
            pre_image = imread(pre_image_batch[b])
            post_image = imread(post_image_batch[b])

            target_image = imread(cls_target_batch[b])
            # Treat "unclassified" as no-damage
            target_image[target_image == 5] = 1

            localization_image = imread(loc_target_batch[b])
            localization_image[localization_image > 1] = 1

            # for CATEGORICAL ONE-HOT classes
            categorical = np.zeros((target_image.shape[0], target_image.shape[1], 5))
            categorical[:, :, 0] = localization_image
            categorical[:, :, 1] = target_image == 1
            categorical[:, :, 2] = target_image == 2
            categorical[:, :, 3] = target_image == 3
            categorical[:, :, 4] = target_image == 4
            target_image = categorical


            # AUGMENTATION TO BE ADDED #
            transform = A.Compose([
                A.RandomCrop(width=self.patch_size, height=self.patch_size, p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(min_holes=4, max_holes=12, max_height=15, max_width=15, mask_fill_value=0, p=0.6),
                A.OneOf([
                    A.MotionBlur(p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.6),
                    A.Blur(blur_limit=3, p=0.6)], p=0.7),
            ], additional_targets={'image_post': 'image'})
            transformed = transform(image=pre_image, image_post=post_image, mask=target_image)

            if self.preprocessing is None:
                x_pre_batch[b] = transformed['image']
                x_post_batch[b] = transformed['image_post']
            else:
                x_pre_batch[b] = self.preprocessing(transformed['image'])
                x_post_batch[b] = self.preprocessing(transformed['image_post'])
            y_target_batch[b] = transformed['mask']

        return (x_pre_batch, x_post_batch), y_target_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)