import torch
import torch.nn as nn
#from torchvision import models
from torch.nn.functional import relu #fonction d'activation
from UNET import *
from gaussian_noise import GaussianNoise
import matplotlib.pyplot as plt
# DATASET
from dataset_loader import load_dataset

batch_size=4
DATASET = load_dataset(batch_size)

# gaussian noise
step=25
sigma = 0.5
gaussian_noise=GaussianNoise(step,sigma=sigma)


# On crée une instance de UNet: 
model = UNet()
parameters=model.parameters()
# Une fois le UNet créé, on définit une fonction de perte : 
loss_fn = torch.nn.MSELoss()
#on va utiliser un optimiseur qui va permettre de minimiser la fonction de perte
optimizer = torch.optim.Adam(parameters, lr=0.0001) #lr est le learning rate (j'ai utilisé celui de l'exemple de PyTorch)





# On crée ensuite une boucle d'entraînement qui itère sur chaque epoch par laquelle passent toutes les données d'entraînement 
num_epochs= 6
counter=0
PATH='./save/modele1'
for epoch in range(num_epochs):
    model.train() # entraîne le modèle
    for img, label in DATASET:
        if int(label[0]):
            optimizer.zero_grad() # s'assure de mettre les gradients à 0 
            noise_image=gaussian_noise(img)
            outputs = model(noise_image)
            loss = loss_fn(img, outputs) #calcule les écarts entre les données du modèle et les données réelles
            loss.backward() # calcule les gradients de la perte par rapport aux paramètres du modèle
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step() # met à jour l'optimisation
            print(loss)
            counter+=1

        if counter%4==1:
            torch.save({
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'loss' : loss,
                }, PATH)
            file = open( "loss", "a" )
            file.write(str(loss.item()) + '\n' )
            file.close()

