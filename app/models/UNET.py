import torch
import torch.nn as nn
#from torchvision import models
from torch.nn.functional import relu #fonction d'activation

# DATASET


class UNet(nn.Module):
    def __init__(self): # On initialise le U-Net
        super().__init__()
        
        # Encodeur : on déifnit les différentes couches
        # Il faut en entrée une image de 572*572 pixels en couleur RGB (3) donc de taille 572*572*3
        # On crée les différentes couches de l'encodeur en suivant le schéma du U-Net
        # Le premier argument indique que l'image est en 3 couleurs, c'est le canal d'entrée
        # Le deuxième argument est le canal de sortie de 64 filtres 
        # taille du noyau est de 3 (voir schéma)
        # padding de 1 pixel autour de l'image d'entrée
        # Premier niveau de convolution: 
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # conv 3*3, image de sortie : 570*570*64
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # conv 3*3, image de sortie : 568*568*64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # max pool, image de sortie : 284*284*64

        # Deuxième niveau de convolution qui prend en entrée une image de 284*284*64
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # conv 3*3, image de sortie : 282*282*128
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # conv 3*3, image de sortie : 280*280*128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # max pool, image de sortie : 140*140*128

        # Troisième niveau qui prend en entrée une image de 140*140*128
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # conv 3*3, image de sortie : 138*138*256
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # conv 3*3, image de sortie : 136*136*256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # max pool, image de sortie : 68*68*256

        # Quatrième niveau qui prend en entrée une image de 68*68*256
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66*66*512
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64*64*512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # max pool, image de sortie : 32*32*512

        # Cinquième niveau qui prend en entrée une image de 32*32*512
        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # conv 3*3, image de sortie : 30*30*1024
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # conv 3*3, image de sortie : 28*28*1024


        # Décodeur : on définit les différentes couches 
        # En entrée, on prend une image de 28*28*1024
        # On va utiliser l'opération de transposée de convolution, qui contrairement à la convolution classique 
        # qui réduit la taille de l'entrée, augmente la taille de l'entrée
        # Premier niveau de déconvolution  
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.deconv1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Deuxième niveau de déconvolution 
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.deconv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Troisième niveau de déconvolution
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.deconv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Quatrième niveau de déconvolution
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1) # 64 canaux d'entrée pour la couche de convolution, 
        # 1 canal de sortie pour cette couche car on cherche à débruiter une image et le noyau vaut 1 car sur 
        # le schéma, on fait une convolution 1*1

#Forward Pass

def forward(self, x):
        # Encodeur
        x11 = relu(self.conv1_1(x))
        x12 = relu(self.conv1_2(x11)) # On va le réutiliser pour un "copy and crop" 
        xp1 = self.pool1(x12)

        x21 = relu(self.conv2_1(xp1))
        x22 = relu(self.conv2_2(x21)) # On va le réutiliser pour un "copy and crop" 
        xp2 = self.pool2(x22)

        x31 = relu(self.conv3_1(xp2))
        x32 = relu(self.conv3_2(x31)) # On va le réutiliser pour un "copy and crop" 
        xp3 = self.pool3(x32)

        x41 = relu(self.conv4_1(xp3))
        x42 = relu(self.conv4_2(x41)) # On va le réutiliser pour un "copy and crop" 
        xp4 = self.pool4(x42)

        x51 = relu(self.conv5_1(xp4))
        x52 = relu(self.conv5_2(x51))
        
        # Decoder
        xup1 = self.upconv1(x52)
        xcrop1 = torch.cat([xup1, x42], dim=1)
        xd11 = relu(self.deconv1_1(xcrop1))
        xd12 = relu(self.deconv1_2(xd11))

        xup2 = self.upconv1(xd12)
        xcrop2 = torch.cat([xup2, x32], dim=1)
        xd21 = relu(self.deconv1_1(xcrop2))
        xd22 = relu(self.deconv1_2(xd21))

        xup3 = self.upconv1(xd22)
        xcrop3 = torch.cat([xup3, x22], dim=1)
        xd31 = relu(self.deconv1_1(xcrop3))
        xd32 = relu(self.deconv1_2(xd31))

        xup4 = self.upconv1(xd32)
        xcrop4 = torch.cat([xup4, x12], dim=1)
        xd41 = relu(self.deconv1_1(xcrop4))
        xd42 = relu(self.deconv1_2(xd41))

        out = self.outconv(xd42)

        return out



#On crée une instance de UNet: 
model = UNet()
parameters=model.parameters()
# Une fois le UNet créé, on définit une fonction de perte : 
loss_fn = torch.nn.MSELoss()
#on va utiliser un optimiseur qui va permettre de minimiser la fonction de perte
optimizer = torch.optim.Adam(parameters, lr=0.001) #lr est le learning rate (j'ai utilisé celui de l'exemple de PyTorch)

# On crée ensuite une boucle d'entraînement qui itère sur chaque epoch par laquelle passent toutes les données d'entraînement 
num_epochs= 2
for epoch in range(num_epochs):
    model.train() # entraîne le modèle
    for images, noise_images in DATASET:
        optimizer.zero_grad() # s'assure de mettre les gradients à 0 
        outputs = model(noise_images)
        loss = loss_fn(outputs, images) #calcule les écarts entre les données du modèle et les données réelles
        loss.backward() # calcule les gradients de la perte par rapport aux paramètres du modèle
        optimizer.step() # met à jour l'optimisation 