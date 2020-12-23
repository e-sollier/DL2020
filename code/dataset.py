import os
from utils import *

class Dataset():
    def __init__(self, 
        tag,
        batch_size=1,
        input_dir= 'data_input', 
        output_dir='data_input',
        random_seed=1996):
        self.batch_size=batch_size
        self.input_dir  = os.path.join(input_dir, tag)
        self.output_dir = os.path.join(output_dir, tag)
        self.X_train  = None
        self.y_train  = None
        self.A_train  = None
        self.Ah_train = None
        self.X_test   = None
        self.y_test   = None
        self.A_test   = None
        self.Ah_test  = None
        self.seed = random_seed

    def create_syn(self, **kwargs):
        self.X_train, self.y_train, self.A_train,\
            self.X_test, self.y_test, self.A_test = gen_syn_data(random_seed=self.seed, **kwargs)

    def load(self, **kwargs):
        self.X_train = load_features(self.input_dir, 'train', **kwargs)
        self.y_train = load_classes(self.input_dir, 'train', **kwargs)
        self.X_test  = load_features(self.input_dir, 'test', **kwargs)
        self.y_test  = load_classes(self.input_dir, 'test', **kwargs)

    def create_graph(self, method='glasso', alphas=5, n_jobs=None):
        #TODO: add **kwargs
        if method == 'glasso':
            self.Ah_train = glasso(self.X_train, alphas, n_jobs)
            self.Ah_test  = glasso(self.X_test, alphas, n_jobs)

    def score_graphs(self):
        # TODO: add error trap
        return compare_graphs(self.A_train, self.Ah_train), \
            compare_graphs(self.A_test, self.Ah_test)

    def comp_test(self):
        # TODO: add error trap
        return compare_graphs(self.Ah_train, self.Ah_test)

    def optim_graphs(self):
        # TODO: add!
        pass

    def _dataloader(self, dataset = 'train',use_true_graph=True):
        if dataset == 'train':
            if use_true_graph:
                return get_dataloader(self.A_train, self.X_train, self.y_train,self.batch_size)
            else:
                return get_dataloader(self.Ah_train, self.X_train, self.y_train,self.batch_size)
        else:
            if use_true_graph:
                return get_dataloader(self.A_train, self.X_test, self.y_test,self.batch_size)
            else:
                return get_dataloader(self.Ah_train, self.X_test, self.y_test,self.batch_size)

    def save(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        file = open("X_train.txt", "w")
        for row in self.X_train:
            np.savetxt(file, row)
        file.close()

        file = open("y_train.txt", "w")
        for row in self.y_train:
            np.savetxt(file, row)
        file.close()

        file = open("adj_train.txt", "w")
        for row in self.A_train:
            np.savetxt(file, row)
        file.close()

        file = open("adjh_train.txt", "w")
        for row in self.Ah_train:
            np.savetxt(file, row)
        file.close()

        file = open("X_test.txt", "w")
        for row in self.X_test:
            np.savetxt(file, row)
        file.close()

        file = open("y_test.txt", "w")
        for row in self.y_test:
            np.savetxt(file, row)
        file.close()

        file = open("adj_test.txt", "w")
        for row in self.A_test:
            np.savetxt(file, row)
        file.close()

        file = open("adjh_test.txt", "w")
        for row in self.Ah_test:
            np.savetxt(file, row)
        file.close()
        
