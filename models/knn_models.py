
import torch

import torch.utils.data
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from joblib import dump, load
from networks.clip import clip

class KnnModel():
    def __init__(self, clip_model='ViT-L/14', ckpt=None, load_encoder=True):
    
        if load_encoder:
            self.clip_model, _ = clip.load(clip_model, device="cpu") 

            print ("CLIP Model loaded..")
            self.clip_model.eval()
            # self.clip_model.to(device)

        self.knn_model = KNeighborsClassifier(n_neighbors=1)
        if ckpt is not None:
            self.knn_model = load(ckpt)

    def train(self, real_features_file, fake_features_file, ckpt=None):
        print('Training the kNN on the training data.')

        real_features = np.load(real_features_file)
        fake_features = np.load(fake_features_file)

        combined_data = np.vstack((real_features, fake_features))
        real_labels = np.zeros(real_features.shape[0])
        fake_labels = np.ones(fake_features.shape[0])
        combined_labels = np.concatenate((real_labels, fake_labels))

        shuffled_indices = np.random.permutation(combined_data.shape[0])
        train_data = combined_data[shuffled_indices]
        train_labels = combined_labels[shuffled_indices]

        print("Fit kNN...")
        self.knn_model.fit(train_data, train_labels)

        if ckpt is not None:
            dump(self.knn_model, ckpt)

    def load_weights(self, ckpt):
        self.knn_model = load(ckpt) 

    def predict(self, img):
        with torch.no_grad():
            features = self.clip_model.encode_image(img).cpu().numpy()  

        predictions = self.knn_model.predict_proba(features)
        return predictions[:, 1]