import torch
import torch.nn as nn
import torchvision


class UNet(nn.Module):
    def __init__(self): # On initialise le U-Net
        super().__init__()
        
        # Encodeur : on déifnit les différentes couches
        # Il faut en entrée une image de 512*512 pixels en couleur RGB (3) donc de taille 512*512*3
        # On crée les différentes couches de l'encodeur en suivant le schéma du U-Net
        # Le premier argument indique que l'image est en 3 couleurs, c'est le canal d'entrée
        # Le deuxième argument est le canal de sortie de 64 filtres 
        # taille du noyau est de 3 (voir schéma)
        # padding de 1 pixel autour de l'image d'entrée : nécessaire ensuite pour le loss

        # Premier niveau de convolution:
        self.layer1 = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3,padding=1), # conv 3*3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU() # conv 3*3
        ])

        self.layer2 = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2), # max pool
            # Deuxième niveau de convolution 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU()
        ])
            
        self.layer3 = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2), # max pool
            # Troisième niveau de convolution 
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU()
        ])


        self.layer4 = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2), # max pool
            # Quatrième niveau de convolution 
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU()
        ])
        

        self.layer5 = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2), # max pool
            # Cinquième niveau de convolution
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), # conv 3*3
            nn.ReLU(),
        # Décodeur : on définit les différentes couches 
        # On va utiliser l'opération de transposée de convolution
        # Premier niveau de déconvolution  
            nn.ConvTranspose2d(1024, 512, kernel_size=2,stride=2)
        ])

        self.layer6 = nn.ModuleList([
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        # Deuxième niveau de déconvolution 
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        ])

        self.layer7 = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        # Troisième niveau de déconvolution
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        ])

        self.layer8 = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        # Quatrième niveau de déconvolution
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])

        self.layer9 = nn.ModuleList([
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        # Couche de sortie
            nn.Conv2d(64, 3, kernel_size=1) 
        ])

    #Forward Pass
    def forward(self, x):

        # Encodeur
        for layer in self.layer1:
            x = layer(x)

        x1 = x.detach().clone() # On va le réutiliser pour un "copy and crop" 
        
        for layer in self.layer2:
            x = layer(x)

        x2 = x.detach().clone() # On va le réutiliser pour un "copy and crop"

        for layer in self.layer3:
            x = layer(x)

        x3 = x.detach().clone() # On va le réutiliser pour un "copy and crop" 
        
        for layer in self.layer4:
            x = layer(x)

        x4 = x.detach().clone() # On va le réutiliser pour un "copy and crop" 
        
        for layer in self.layer5:
            x = layer(x)

        #Décodeur :
        x = torch.cat((x, x4), dim=1)
        for layer in self.layer6:
            x = layer(x)

        x = torch.cat((x,x3), dim=1)
        for layer in self.layer7:
            x = layer(x)

        x = torch.cat((x,x2), dim=1)
        for layer in self.layer8:
            x = layer(x)

        x = torch.cat((x,x1), dim=1)
        for layer in self.layer9:
            x = layer(x)

        return x
