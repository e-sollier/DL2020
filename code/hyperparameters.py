from classifier import Classifier

def select_hyperparameters_CV(
    dataset,
    n_features,
    n_classes,
    n_hidden_GNN=[10],
    n_hidden_FC=[],
    K=4,
    classifier='MLP', 
    lr=.01, 
    momentum=.9,
    epochs=50,
    device='cpu',
    batch_size=16):
    """
    Select the best dropout rate using cross-validation
    """
    best_rate=0
    best_score=0
    for dropout_rate in [0,0.1,0.2,0.5]:
        score=0
        for train_dataloader,val_dataloader in dataset.CV_dataloaders(use_true_graph=True,n_splits=5,batch_size=batch_size):
            
            clf = Classifier(n_features=n_features,n_classes=n_classes,classifier=classifier,K=K,n_hidden_FC=n_hidden_FC,n_hidden_GNN=n_hidden_GNN,\
                dropout_GNN = dropout_rate, dropout_FC=dropout_rate, lr=lr,momentum=momentum,device=device)

            clf.fit(train_dataloader, epochs = epochs, test_dataloader=val_dataloader,verbose=False)
            score+= clf.eval(val_dataloader,verbose=False)[0]

        if score>best_score:
            best_score = score
            best_rate = dropout_rate

    return best_rate

