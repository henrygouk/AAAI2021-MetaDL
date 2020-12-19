""" This is a dummy baseline. It is just supposed to check if ingestion and 
scoring are called properly.
"""
import os
import logging
import csv

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import torch
import torchvision
import numpy as np

from metadl.api.api import MetaLearner, Learner, Predictor

from torch.autograd import Function

os.environ["TORCH_HOME"] = '/app/codalab'

class PLogDet(Function):
    @staticmethod
    def forward(ctx, l, l_inv):
        ctx.save_for_backward(l_inv)
        return 2 * l.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    @staticmethod
    def backward(ctx, g):
        l_inv, = ctx.saved_tensors
        return g * l_inv, None

plogdet = PLogDet.apply

to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))

def metadataset_to_torch(dataset):
    for i, (e, _) in enumerate(dataset):
        yield (
                (to_torch_imgs(e[0]), to_torch_labels(e[1])),
                (to_torch_imgs(e[3]), to_torch_labels(e[4]))
            )

class MyMetaLearner(MetaLearner):

    def __init__(self):
        super().__init__()

    def meta_fit(self, meta_dataset_generator) -> Learner:
        """
        Args:
            meta_dataset_generator : a DataGenerator object. We can access
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.

        Returns:
            MyLearner object : a Learner that stores the meta-learner's
                learning object. (e.g. a neural network trained on meta-train
                episodes)
        """
        meta_train_dataset = metadataset_to_torch(meta_dataset_generator.meta_train_pipeline)
        meta_valid_dataset = metadataset_to_torch(meta_dataset_generator.meta_valid_pipeline)

        self.feature_dim = 512
        extractor = torchvision.models.resnet18(pretrained=True)
        extractor.fc = torch.nn.Identity()
        extractor.cuda()

        m = torch.nn.Parameter(torch.zeros((self.feature_dim,)).cuda())
        S_lower = torch.nn.Parameter(torch.eye(self.feature_dim).cuda())
        kappa = torch.nn.Parameter(torch.tensor(float(1)).cuda())
        nu = torch.nn.Parameter(torch.tensor(float(self.feature_dim)).cuda())

        learner = MyLearner(extractor, m, S_lower, kappa, nu)

        from itertools import chain
        optimiser = torch.optim.Adam(chain(extractor.parameters(), [m, S_lower, kappa, nu]), lr=0.0001)
        loss_fn = torch.nn.CrossEntropyLoss()

        extractor.train()

        running_loss = 0
        running_acc = 0
        ctr = 0
        print_freq = 50
        max_steps = 10000

        for (train_images, train_labels), (val_images, val_labels) in meta_train_dataset:
            train_images = train_images.cuda()
            train_labels = train_labels.cuda()
            val_images = val_images.cuda()
            val_labels = val_labels.cuda()

            optimiser.zero_grad()
            predictor = learner.fit_episode(train_images, train_labels)
            val_preds = predictor.predict_episode(val_images)
            val_loss = loss_fn(val_preds, val_labels)
            val_loss.backward()
            optimiser.step()

            with torch.no_grad():
                kappa.copy_(torch.clamp(kappa, min=0))
                nu.copy_(torch.clamp(nu, min=self.feature_dim))
                val_acc = torch.sum(val_preds.max(dim=1)[1] == val_labels).item() / val_preds.shape[0]

            running_loss += val_loss.item()
            running_acc += val_acc
            ctr += 1

            if ctr % print_freq == 0:
                print("{:06d}\t\t{:2.6f}\t\t{:.6f}".format(int(ctr / print_freq), running_loss / print_freq, running_acc / print_freq))
                running_loss = 0
                running_acc = 0
                optimiser.param_groups[0]['lr'] = 0.0001 * (0.98 ** (ctr / print_freq))

            if ctr == max_steps:
                break

        return learner

class MyLearner(Learner):

    def __init__(self, extractor=None, m=None, S_lower=None, kappa=None, nu=None, num_classes=5):
        super().__init__()
        self.extractor = extractor
        self.m = m
        self.S_lower = S_lower
        self.kappa = kappa
        self.nu = nu
        self.num_classes = num_classes

    def fit_episode(self, images, labels):
        """
            Note: this method assumes 1-shot
        """
        # Set up the place where we will put the parameters for each class
        mus = [None for i in range(self.num_classes)]
        scales = [None for i in range(self.num_classes)]

        features = self.extractor(images)

        def self_outer(v):
            return torch.mm(v.unsqueeze(1), v.unsqueeze(0))

        kappa_N = self.kappa + 1
        nu_N = self.nu + 1
        kappa_m_outer = self.kappa * self_outer(self.m)
        d = features.shape[1]
        self.S = torch.mm(self.S_lower, self.S_lower.t())

        # Compute the parameters for each class
        for fs, lbl in zip(features, labels):
            m_N = (self.kappa * self.m + fs) / kappa_N
            S_N = self.S + self_outer(fs) + kappa_m_outer - kappa_N * self_outer(m_N)
            mus[lbl] = m_N
            scales[lbl] = (kappa_N + 1) / (kappa_N * (nu_N - d + 1)) * S_N

        return MyPredictor(self.extractor, mus, scales, nu_N - d + 1)

    def fit(self, dataset_train) -> Predictor:
        """
        Args:
            dataset_train : a tf.data.Dataset object. It is an iterator over
                the support examples.
        Returns:
            ModelPredictor : a Predictor.
        """
        for images, labels in dataset_train:
            return self.fit_episode(to_torch_imgs(images).cuda(), to_torch_labels(labels).cuda())

    def save(self, model_dir):
        """ Saves the learning object associated to the Learner. It could be
        a neural network for example.

        Note : It is mandatory to write a file in model_dir. Otherwise, your
        code won't be available in the scoring process (and thus it won't be
        a valid submission).
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))

        def save_object(obj, name):
            torch.save(obj, os.path.join(model_dir, name))

        save_object(self.extractor.state_dict(), "extractor.pt")
        save_object(self.m, "m.pt")
        save_object(self.S_lower, "S_lower.pt")
        save_object(self.kappa, "kappa.pt")
        save_object(self.nu, "nu.pt")

    def load(self, model_dir):
        """ Loads the learning object associated to the Learner. It should
        match the way you saved this object in save().
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))

        def load_object(name):
            return torch.load(os.path.join(model_dir, name))

        self.extractor = torchvision.models.resnet18()
        self.extractor.fc = torch.nn.Identity()
        self.extractor.load_state_dict(load_object("extractor.pt"))
        self.extractor.cuda()
        self.extractor.eval()

        self.m = load_object("m.pt").cuda()
        self.S_lower = load_object("S_lower.pt").cuda()
        self.kappa = load_object("kappa.pt").cuda()
        self.nu = load_object("nu.pt").cuda()

class MyPredictor(Predictor):

    def __init__(self, extractor, mus, scales, df):
        super().__init__()
        d = mus[0].shape[0]
        self.extractor = extractor
        self.mus = mus
        self.df = df
        self.biases = []
        self.d = d

        cholesky_factors = [torch.cholesky(s) for s in scales]
        self.inv_scales = [torch.cholesky_inverse(c) for c in cholesky_factors]

        for c, inv in zip(cholesky_factors, self.inv_scales):
            b = torch.lgamma(0.5 * (df + d)) - torch.lgamma(0.5 * df) - 0.5 * d * torch.log(df) - plogdet(c, inv)
            self.biases.append(b)

    def predict_episode(self, images):
        fs = self.extractor(images)
        total = 0.0
        preds = torch.zeros((images.shape[0], len(self.mus))).cuda()

        for i in range(fs.shape[0]):
            for c, (mu, inv_scale, b) in enumerate(zip(self.mus, self.inv_scales, self.biases)):
                diff = fs[i] - mu
                dist = torch.mm(torch.mm(diff.unsqueeze(0), inv_scale), diff.unsqueeze(1))
                preds[i, c] = b - 0.5 * (self.df + self.d) * torch.log(1.0 + (1.0 / self.df) * dist)

        return preds

    def predict(self, dataset_test):
        """ Predicts the label of the examples in the query set which is the 
        dataset_test in this case. The prototypes are already computed by
        the Learner.

        Args:
            dataset_test : a tf.data.Dataset object. An iterator over the 
                unlabelled query examples.
        Returns: 
            preds : tensors, shape (num_examples, N_ways). We are using the 
                Sparse Categorical Accuracy to evaluate the predictions. Valid 
                tensors can take 2 different forms described below.

        Case 1 : The i-th prediction row contains the i-th example logits.
        Case 2 : The i-th prediction row contains the i-th example 
                probabilities.

        Since in both cases the SparseCategoricalAccuracy behaves the same way,
        i.e. taking the argmax of the row inputs, both forms are valid.
        Note : In the challenge N_ways = 5 at meta-test time.
        """
        for images in dataset_test:
            return tf.convert_to_tensor(self.predict_episode(to_torch_imgs(images[0]).cuda()).detach().cpu().numpy())

