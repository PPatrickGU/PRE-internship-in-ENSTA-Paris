"""

Architectures of different neural models to solve the problem

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math


#models of Nazar: Form line 17 to line 782

class AutoEncoder(nn.Module):
    def __init__(self, cuda = True, num_channel=3, h_dim=2688, z_dim=1024):
        super(AutoEncoder, self).__init__()
        self.cuda_p = cuda
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )
        # self.fc0 = nn.Linear(h_dim, int(h_dim/2))
        # self.dropout0 = nn.Dropout(p=0.1)
        # self.fc00 = nn.Linear(int(h_dim/2), int(h_dim/2))
        # self.dropout00 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)


        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding = 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding= 1, output_padding = 0),
            # nn.BatchNorm2d(3),
            nn.Tanh()
        )


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())   #without the * the same thing?
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu, logvar = self.fc1(h), F.relu(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 32, 7, 12)
        z = self.decoder(z)
        return z


    def forward(self, x):
        features = self.encode(x)
        z = self.decode(features)
        return features, z


class CNN_stack_FC_first(nn.Module):
    def __init__(self, cuda = True, num_channel = 3,  cnn_fc_size = 1024, num_output=20):
        super(CNN_stack_FC_first, self).__init__()
        self.cuda_p = cuda
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )
        self.fc1 = nn.Linear(5376  , cnn_fc_size) #5376 / 20736
        self.fc2 = nn.Linear(cnn_fc_size, 128)
        self.fc3 = nn.Linear(128, num_output)
        self.dropout0 = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p= 0.4)


    def forward(self, x, p_and_roll, num_images):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x =  torch.tanh(self.fc3(x)).view(x.size(0), -1, 2)

        return x


class CNN_stack_FC(nn.Module):
    def __init__(self, cuda = True, num_channel = 3,  cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_stack_FC, self).__init__()
        self.cuda_p = cuda
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.dropout0 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(5376  , cnn_fc_size) #5376 / 20736

        self.fc2 = nn.Linear(cnn_fc_size, 512) #5376 / 20736
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_output)
        self.dropout1 = nn.Dropout(p=0.3)

        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p= 0.4)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu, logvar = self.mu(h), F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def forward(self, x, p_and_roll, num_images):

        x = self.encode(x).view(x.size(0), 1, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x)).view(x.size(0), -1, 2)

        return x


class CNN_stack_PR_FC(nn.Module):
    def __init__(self, cuda = True, num_channel = 3,  cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_stack_PR_FC, self).__init__()
        self.cuda_p = cuda
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),  # 8x54x96
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),              # 8x27x48
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),           # 16x27x48
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),                 # 16x14x24
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 32x14x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),                 # 32x7x12
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.dropout0 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(5376  , 1024) #5376 / 20736

        self.fc2 = nn.Linear(cnn_fc_size, 1024) #5376 / 20736
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, num_output)
        self.dropout1 = nn.Dropout(p=0.3)

        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p= 0.4)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu, logvar = self.mu(h), F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def forward(self, x, p_and_roll, num_images):

        x = self.encode(x).view(x.size(0), 1, -1)

        PR = [p_and_roll[:,i,:] for i in range(num_images-1, -1, -1)]
        PR = torch.cat(PR, 1).view(x.size(0), 1, -1)

        # input_fc = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(num_images)]
        input_fc = torch.cat((x, PR), 2).view(x.size(0), 1, -1)

        x = F.relu(self.fc2(input_fc))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x)).view(x.size(0), -1, 2)

        return x


class CNN_PR_FC (nn.Module):
    def __init__(self, cuda = True, cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_PR_FC, self).__init__()
        self.cuda_p = cuda
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.dropout0 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(5376  , 1024) #5376 / 20736

        self.fc2 = nn.Linear(cnn_fc_size, int(cnn_fc_size/2)) #5376 / 20736
        self.fc22 = nn.Linear(int(cnn_fc_size/2), 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, num_output)
        self.dropout1 = nn.Dropout(p=0.3)

        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p= 0.4)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu, logvar = self.mu(h), F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def forward(self, x, p_and_roll, num_images):

        features = [self.encode(x[:,i,:,:,:]) for i in range(num_images-1, -1, -1)]

        PR = [p_and_roll[:,i,:] for i in range(num_images-1, -1, -1)]

        input_fc = [ torch.cat((features[i], PR[i]), 1).view(x.size(0), 1, -1) for i in range(num_images)]
        input_fc = torch.cat(input_fc, 2).view(x.size(0), 1, -1)

        x = F.relu(self.fc2(input_fc))
        x = self.dropout2(x)
        x = F.relu(self.fc22(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x)).view(x.size(0), -1, 2)

        return x


class LSTM_encoder_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)
        #print(encoder_output - encoder_hidden[0].permute(1,0,2))
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1) # (12,10,2)

        return decoder_output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_decoder_images_PR (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]

        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)



        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)

        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_decoder_images(nn.Module):
    def __init__(self,cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 1024, decoder_hidden_size = 1024,  output_size = 20):
        super(CNN_LSTM_encoder_decoder_images, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]

        lstm_input_features = torch.cat(features, 1).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden

class CNN_LSTM_image_encoder_PR_encoder_decoder(nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, im_encoder_input_size = 4096, pr_encoder_input_size = 20 , im_encoder_hidden_size = 128, pr_encoder_hidden_size = 128, decoder_hidden_size = 256,  output_size = 20):
        super(CNN_LSTM_image_encoder_PR_encoder_decoder, self).__init__()
        self.cuda_p = cuda
        self.im_encoder_hidden_size = im_encoder_hidden_size
        self.pr_encoder_hidden_size = pr_encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.im_encoder_lstm = nn.LSTM(im_encoder_input_size, im_encoder_hidden_size, batch_first=True)
        self.pr_encoder_lstm = nn.LSTM(pr_encoder_input_size, pr_encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_hidden_size, int(decoder_hidden_size/2), batch_first=True)

        self.decoder_fc_1 = nn.Linear(int(decoder_hidden_size/2), int(decoder_hidden_size/4))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/4), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoderIm(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.im_encoder_hidden_size)


    def initHiddenEncoderPR(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.pr_encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,int(self.decoder_hidden_size/2))


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, im_encoder_hidden, pr_encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = torch.cat(features, 1).view(image_s[0].size(0), 1, -1)
        lstm_input_PR = torch.cat(PR, 1).view(image_s[0].size(0), 1, -1)

        encoder_output_images, im_encoder_hidden = self.im_encoder_lstm(lstm_input_features,  im_encoder_hidden )
        encoder_output_PR, pr_encoder_hidden = self.pr_encoder_lstm(lstm_input_PR,  pr_encoder_hidden )

        lstm_input_decoder = torch.cat((encoder_output_images, encoder_output_PR), 2).view(image_s[0].size(0), 1, -1)
        decoder_output, decoder_hidden = self.LSTM_decoder(lstm_input_decoder, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, im_encoder_hidden, pr_encoder_hidden, decoder_hidden


class CNN_LSTM_decoder_images_PR(nn.Module):
    def __init__(self,cuda = True, h_dim=2688, z_dim=1024, decoder_input_size = 1000, decoder_hidden_size = 1000,  output_size = 20, drop_par = 0.2):
        super(CNN_LSTM_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = 1
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)
        self.dropout0 = nn.Dropout(p=drop_par)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = self.dropout0(outputs)
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)


        decoder_output, decoder_hidden = self.LSTM_decoder(lstm_input_features, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden



#######################################################
#Models of Dajing
class LSTM_encoder_GRU_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_GRU_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)
        self.decoder_gru = nn.GRU(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def GRU_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_gru(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)
        #print(encoder_output - encoder_hidden[0].permute(1,0,2))
        decoder_output, decoder_hidden = self.GRU_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1) # (12,10,2)

        return decoder_output, encoder_hidden, decoder_hidden



class LSTM_encoder_GRU_decoder_PR_many(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_GRU_decoder_PR_many, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)
        self.decoder_gru = nn.GRU(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def GRU_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_gru(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), use_n_im, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)
        #print(encoder_output - encoder_hidden[0].permute(1,0,2))
        decoder_output, decoder_hidden = self.GRU_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1) # (12,10,2)

        return decoder_output, encoder_hidden, decoder_hidden


class GRU_encoder_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(GRU_encoder_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_gru = nn.GRU(encoder_input_size, encoder_hidden_size, batch_first= True)
        self.decoder_gru = nn.GRU(decoder_hidden_size, decoder_hidden_size,batch_first= True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def GRU_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_gru(inputs, hiddens)
        return outputs, hiddens


    def GRU_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_gru(outputs, hiddens)

        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return torch.zeros(1, n_batch ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):
        PR  = [pr_s[i] for i in range(use_n_im)]

        gru_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.GRU_encoder(gru_input_features, encoder_hidden)

        decoder_output, decoder_hidden = self.GRU_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden

class GRU_encoder_decoder_PR_many(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(GRU_encoder_decoder_PR_many, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_gru = nn.GRU(encoder_input_size, encoder_hidden_size, batch_first= True)
        self.decoder_gru = nn.GRU(decoder_hidden_size, decoder_hidden_size,batch_first= True)

        self.decoder_fc_0 = nn.Linear(output_size, decoder_hidden_size)
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def GRU_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_gru(inputs, hiddens)
        return outputs, hiddens


    def GRU_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_gru(outputs, hiddens)

        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return torch.zeros(1, n_batch ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]
        gru_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), use_n_im, -1) #(24,10,2)
        encoder_output, encoder_hidden = self.GRU_encoder(gru_input_features, encoder_hidden)

        if self.cuda_p:
             output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
        else:
             output = torch.zeros(pr_s[0].size(0), 1, 2)

        decoder_input = torch.tanh(self.decoder_fc_0(gru_input_features[:,-1,:].unsqueeze(1)))
        decoder_hidden = encoder_hidden
        for t in range(0, predict_n_pr):
            #decoder_output = [batch_size, 1, output_size = 2]
            decoder_output, decoder_hidden = self.GRU_decoder(decoder_input, decoder_hidden)
            output = torch.cat([output, decoder_output], 1)
        #output = [batchize, predict_n_pr+1, output_size = 2]
        output = output[:,1:,: ]
        #output = [batch_size, predict_n_pr, output_size = 2]
        return output, encoder_hidden, decoder_hidden


class GRU_encoder_attention_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, batch_size = 12, decoder_hidden_size = 300,  output_size = 20):
        super(GRU_encoder_attention_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        #self.sequence_length = 1


        self.encoder_gru = nn.GRU(encoder_input_size, encoder_hidden_size, batch_first=True)     #batch_first only influences the input, not the hidden

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

        self.decoder_gru = nn.GRU(encoder_hidden_size * 2, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def GRU_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_gru(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]
        s = s.repeat(seq_len, 1, 1)

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0,1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)

    def GRU_decoder(self, decoder_input, s, encoder_output):

        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]
        a = self.attention_net(s, encoder_output).transpose(1,2)

        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)

        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]

        decoder_input = torch.cat((decoder_input, c), dim=2)

        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        #decoder_hidden = [s, c.transpose(0,1)] in LSTM
        decoder_hidden = s #in GRU

        decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)

        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c, encoder_output), dim=2)

        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden

    def initHiddenEncoder(self, n_batch):
        return torch.zeros( 1, n_batch , self.encoder_hidden_size)

    def initHiddenDecoder(self, n_batch):
        return torch.zeros( 1 ,n_batch ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]   #pr_s[0].size(0)=batch_size
        gru_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)  # batchsize, seq_length = 1, output_szie
        #encoder_output = [batchsize, seq_length = 1, encoder_hidden_size]
        encoder_output, encoder_hidden = self.GRU_encoder(gru_input_features, encoder_hidden)

        #s = encoder_hidden[0][-1,:,:].unsqueeze(0) # in LSTM
        s = encoder_hidden[-1,:,:].unsqueeze(0)

        decoder_input = encoder_output[:,-1,:].unsqueeze(1)

        decoder_output, decoder_hidden = self.GRU_decoder(decoder_input, s, encoder_output) #[24, 1, 20]

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1) #[24, 10, 2]

        return decoder_output, encoder_hidden, decoder_hidden

class GRU_encoder_attention_decoder_PR_many(nn.Module):
        def __init__(self, cuda=True, encoder_input_size=10, encoder_hidden_size=300, decoder_hidden_size=300,
                     output_size=20):
            super(GRU_encoder_attention_decoder_PR_many, self).__init__()
            self.cuda_p = cuda
            self.encoder_hidden_size = encoder_hidden_size
            self.decoder_hidden_size = decoder_hidden_size

            self.encoder_gru = nn.GRU(encoder_input_size, encoder_hidden_size, batch_first=True)

            self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
            self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

            self.decoder_gru = nn.GRU(decoder_hidden_size * 2, decoder_hidden_size, batch_first=True)

            self.decoder_fc_0 = nn.Linear(output_size, decoder_hidden_size) #for the decoder_input
            self.decoder_fc_1 = nn.Linear(decoder_hidden_size + encoder_hidden_size , int(decoder_hidden_size / 2))
            self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size / 2), output_size)

        def GRU_encoder(self, inputs, hiddens):
            outputs, hiddens = self.encoder_gru(inputs, hiddens)
            return outputs, hiddens

        def attention_net(self, s, encoder_output):
            # s = [1, batch_size, decoder_hidden_size]
            # encoder_output = [batch_size, seq_len, encoder_hidden_size]

            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]

            # repeat decoder hidden state seq_length times
            # s = [seq_len, batch_size, dedcoder_hidden_size]
            # encoder_output = [batch_size, seq_length, encoder_hidden_size]
            s = s.repeat(seq_len, 1, 1)

            # energy = [batch_size, seq_len, decoder_hiden_size]
            energy = torch.tanh(self.attn(torch.cat((s.transpose(0, 1), encoder_output), dim=2)))

            # attention = [batch_size, seq_len, 1]
            attention = self.v(energy)

            return F.softmax(attention, dim=1)

        def GRU_decoder(self, decoder_input, s, encoder_output):
            # decoder_input = [batch_size, 1, output_size = 20]
            # s = [1, batch_size, decoder_hidden_size]
            # encoder_output = [batch_size, seq_length = 10, encoder_hidden_size]
            # a = [batch_size, 1, seq_len]
            a = self.attention_net(s, encoder_output).transpose(1, 2)

            # c = [batch_size, 1, encoder_hidden_size]
            c = torch.bmm(a, encoder_output)

            # decoder_input = [batch_size, seq_len = 1, encoder_hidden_size * 2]

            decoder_input = torch.cat((decoder_input, c), dim=2)

            # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
            # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

            # decoder_hidden = [s, c.transpose(0,1)] in LSTM
            decoder_hidden = s  # in GRU


            decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)

            # encoder_output = [batch_size, 1, encoder_hidden_size]
            # c = [batch_size, 1, encoder_hidden_size]

            # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
            outputs = torch.cat((decoder_output, c), dim=2)


            # outputs = [batch_size, 1, output_size = 20]
            outputs = F.relu(self.decoder_fc_1(outputs))
            outputs = torch.tanh(self.decoder_fc_2(outputs))

            return outputs, decoder_hidden

        def initHiddenEncoder(self, n_batch):
            return torch.zeros(1, n_batch, self.encoder_hidden_size)

        def initHiddenDecoder(self, n_batch):
            return torch.zeros(1, n_batch, self.decoder_hidden_size)

        def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

            PR = [pr_s[i] for i in range(use_n_im)]
            gru_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), use_n_im, -1)  # (24,10,2)
            encoder_output, encoder_hidden = self.GRU_encoder(gru_input_features, encoder_hidden)

            if self.cuda_p:
                output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
            else:
                output = torch.zeros(pr_s[0].size(0), 1, 2)

            decoder_input = torch.tanh(self.decoder_fc_0(gru_input_features[:, -1, :].unsqueeze(1)))

            s = encoder_hidden[-1, :, :].unsqueeze(0) #[1, batch_size, encoder_hidden_size]

            for t in range(0, predict_n_pr):
                # decoder_output = [batch_size, 1, output_size = 2]

                decoder_output, s = self.GRU_decoder(decoder_input, s, encoder_output)

                output = torch.cat([output, decoder_output], 1)

                top1 = decoder_output.argmax(1).float().unsqueeze(1)

                decoder_input = torch.tanh(self.decoder_fc_0(top1))

            # output = [batchize, predict_n_pr +1 , output_size = 2]
            output = output[:, 1:, :]
            decoder_hidden = s
            # output = [batch_size, predict_n_pr, output_size = 2]

            return output, encoder_hidden, decoder_hidden

class LSTM_encoder_attention_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, batch_size = 12, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_attention_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        #self.sequence_length = 1


        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)     #batch_first only influences the input, not the hidden

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)


        self.decoder_lstm = nn.LSTM(encoder_hidden_size * 2, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(encoder_hidden_size + decoder_hidden_size*2, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]
        s = s.repeat(seq_len, 1, 1)

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0,1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)


    def LSTM_decoder(self, decoder_input, s, encoder_output):

        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]

        a = self.attention_net(s, encoder_output).transpose(1,2)

        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)

        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]

        decoder_input = torch.cat((decoder_input, c), dim=2)

        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        decoder_hidden = [s, c.transpose(0,1)]

        decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)

        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c, encoder_output), dim=2)

        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden

    def initHiddenEncoder(self, n_batch):
        return torch.zeros( 1, n_batch , self.encoder_hidden_size)

    def initHiddenDecoder(self, n_batch):
        return torch.zeros( 1 ,n_batch ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]   #pr_s[0].size(0)=batch_size
        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)  # batchsize, seq_length = 1, output_szie

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)

        s = encoder_hidden[0][-1,:,:].unsqueeze(0)
        decoder_input = encoder_output[:,-1,:].unsqueeze(1)

        decoder_output, decoder_hidden = self.LSTM_decoder(decoder_input, s, encoder_output) #[24, 1, 20]

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1) #[24, 10, 2]

        return decoder_output, encoder_hidden, decoder_hidden


class LSTM_encoder_GRU_attention_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, batch_size = 12, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_GRU_attention_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        #self.sequence_length = 1


        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)     #batch_first only influences the input, not the hidden

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)


        self.decoder_gru = nn.GRU(encoder_hidden_size * 2, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]
        s = s.repeat(seq_len, 1, 1)

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0,1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)


    def LSTM_decoder(self, decoder_input, s, encoder_output):

        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]

        a = self.attention_net(s, encoder_output).transpose(1,2)

        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)

        # decoder_input = [batch_size, seq_len = 1, encoder_hidden_size * 2]

        decoder_input = torch.cat((decoder_input, c), dim=2)

        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        #decoder_hidden = [s, c.transpose(0,1)] #in LSTM
        decoder_hidden = s #in GRU

        decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)

        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c, encoder_output), dim=2)

        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden

    def initHiddenEncoder(self, n_batch):
        return torch.zeros( 1, n_batch , self.encoder_hidden_size)

    def initHiddenDecoder(self, n_batch):
        return torch.zeros( 1 ,n_batch ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]   #pr_s[0].size(0)=batch_size
        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)  # [batchsize, seq_length = 1, output_szie]
        #encoder_output = [batchsize, seq_length = 1, encoder_hidden_size]
        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)

        s = encoder_hidden[0][-1,:,:].unsqueeze(0)
        decoder_input = encoder_output[:,-1,:].unsqueeze(1)

        decoder_output, decoder_hidden = self.LSTM_decoder(decoder_input, s, encoder_output) #[24, 1, 20]

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1) #[24, 10, 2]

        return decoder_output, encoder_hidden, decoder_hidden


class LSTM_encoder_decoder_PR_many(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_decoder_PR_many, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first= True)
        self.decoder_lstm = nn.LSTM(decoder_hidden_size, decoder_hidden_size,batch_first= True)

        self.decoder_fc_0 = nn.Linear(output_size, decoder_hidden_size)
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)

        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return torch.zeros(1, n_batch ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR  = [pr_s[i] for i in range(use_n_im)]
        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), use_n_im, -1) #(24,10,2)


        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)


        if self.cuda_p:
             output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
        else:
             output = torch.zeros(pr_s[0].size(0), 1, 2)

        decoder_input = torch.tanh(self.decoder_fc_0(lstm_input_features[:,-1,:].unsqueeze(1)))
        decoder_hidden = encoder_hidden


        for t in range(0, predict_n_pr):
            #decoder_output = [batch_size, 1, output_size = 2]
            decoder_output, decoder_hidden = self.LSTM_decoder(decoder_input,decoder_hidden)
            output = torch.cat([output, decoder_output], 1)
        #output = [batchize, 11, output_size = 2]
        output = output[:,1:,: ]
        #output = [batch_size, 10, output_size = 2]
        return output, encoder_hidden, decoder_hidden


class LSTM_encoder_attention_decoder_PR_many(nn.Module):
    def __init__(self, cuda=True, encoder_input_size=10, encoder_hidden_size=300, decoder_hidden_size=300,
                 output_size=20):
        super(LSTM_encoder_attention_decoder_PR_many, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

        self.decoder_lstm = nn.LSTM(decoder_hidden_size * 2, decoder_hidden_size, batch_first=True)

        self.decoder_fc_0 = nn.Linear(output_size, decoder_hidden_size)  # for the decoder_input
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size + encoder_hidden_size, int(decoder_hidden_size / 2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size / 2), output_size)

    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]

        s = s[0].repeat(seq_len, 1, 1)

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0, 1), encoder_output), dim=2)))


        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)

    def LSTM_decoder(self, decoder_input, s, encoder_output):
        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]
        a = self.attention_net(s, encoder_output).transpose(1, 2)


        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)


        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]
        decoder_input = torch.cat((decoder_input, c), dim=2)

        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        decoder_hidden = s

        decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)

        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c), dim=2)

        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden

    def initHiddenEncoder(self, n_batch):
        return torch.zeros(1, n_batch, self.encoder_hidden_size)

    def initHiddenDecoder(self, n_batch):
        return torch.zeros(1, n_batch, self.decoder_hidden_size)

    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        PR = [pr_s[i] for i in range(use_n_im)]
        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), use_n_im, -1)  # (24,10,2)
        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)

        if self.cuda_p:
            output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
        else:
            output = torch.zeros(pr_s[0].size(0), 1, 2)

        decoder_input = torch.tanh(self.decoder_fc_0(lstm_input_features[:, -1, :].unsqueeze(1)))

        # s = encoder_hidden[-1, :, :].unsqueeze(0) # for GRU
        s = encoder_hidden  # for LSTM

        for t in range(0, predict_n_pr):
            # decoder_output = [batch_size, 1, output_size = 2]

            decoder_output, s = self.LSTM_decoder(decoder_input, s, encoder_output)

            output = torch.cat([output, decoder_output], 1)
            top1 = decoder_output.argmax(1).float().unsqueeze(1)
            decoder_input = torch.tanh(self.decoder_fc_0(top1))

        # output = [batchize, predict_n_pr +1 , output_size = 2]
        output = output[:, 1:, :]
        decoder_hidden = s
        # output = [batch_size, predict_n_pr, output_size = 2]

        return output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_decoder_images_PR_many (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_decoder_images_PR_many, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_0 = nn.Linear(1026, decoder_hidden_size)
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]

        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0),use_n_im , -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)

        if self.cuda_p:
             output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
        else:
             output = torch.zeros(pr_s[0].size(0), 1, 2)

        decoder_input = torch.tanh(self.decoder_fc_0(lstm_input_features[:,-1,:].unsqueeze(1)))
        decoder_hidden = encoder_hidden


        for t in range(0, predict_n_pr):
            #decoder_output = [batch_size, 1, output_size = 2]
            decoder_output, decoder_hidden = self.LSTM_decoder(decoder_input,decoder_hidden)
            output = torch.cat([output, decoder_output], 1)
        #output = [batchize, 11, output_size = 2]
        output = output[:,1:,: ]
        #output = [batch_size, 10, output_size = 2]
        return output, encoder_hidden, decoder_hidden

class CNN_LSTM_encoder_GRU_decoder_images_PR (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_GRU_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_gru = nn.GRU(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def GRU_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_gru(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        #print(len(features))
        #print(len(features[0][0]))
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 1).view(image_s[0].size(0), use_n_im, -1)

        if self.cuda_p:
             output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
        else:
             output = torch.zeros(pr_s[0].size(0), 1, 2)

        decoder_input = torch.tanh(self.decoder_fc_0(lstm_input_features[:,-1,:].unsqueeze(1)))
        decoder_hidden = encoder_hidden


        for t in range(0, predict_n_pr):
            #decoder_output = [batch_size, 1, output_size = 2]
            decoder_output, decoder_hidden = self.LSTM_decoder(decoder_input,decoder_hidden)
            output = torch.cat([output, decoder_output], 1)
        #output = [batchize, 11, output_size = 2]
        output = output[:,1:,: ]
        #output = [batch_size, 10, output_size = 2]
        return output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_attention_decoder_images_PR (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 600, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_attention_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

        self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_0 = nn.Linear(encoder_input_size, decoder_hidden_size)
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size + encoder_hidden_size*2, int(decoder_hidden_size / 2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size / 2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]
        s = s.repeat(seq_len, 1, 1)

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0,1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)


    def LSTM_decoder(self, decoder_input, s, encoder_output):

        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]

        a = self.attention_net(s, encoder_output).transpose(1,2)

        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)
        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]

        decoder_input = torch.cat((decoder_input, c), dim=2)
        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        decoder_hidden = [s, c.transpose(0,1)]



        decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c, encoder_output), dim=2)
        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden

    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        #print(len(features))
        #print(len(features[0][0]))
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)


        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)


        s = encoder_hidden[0][-1, :, :].unsqueeze(0)

        #decoder_input = encoder_output[:,-1,:].unsqueeze(1)
        decoder_input = torch.tanh(self.decoder_fc_0(lstm_input_features[:, -1, :].unsqueeze(1)))

        decoder_output, decoder_hidden = self.LSTM_decoder(decoder_input, s, encoder_output)  # [batch_size, 1, output_size]

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_GRU_attention_decoder_images_PR (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 600, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_GRU_attention_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

        self.decoder_gru = nn.GRU(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_0 = nn.Linear(encoder_input_size, decoder_hidden_size)
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size + encoder_hidden_size*2, int(decoder_hidden_size / 2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size / 2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]
        s = s.repeat(seq_len, 1, 1)

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0,1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)


    def LSTM_decoder(self, decoder_input, s, encoder_output):

        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]

        a = self.attention_net(s, encoder_output).transpose(1,2)


        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)
        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]


        decoder_input = torch.cat((decoder_input, c), dim=2)
        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        #decoder_hidden = [s, c.transpose(0,1)] #in LSTM
        decoder_hidden = s  # in GRU

        decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c, encoder_output), dim=2)
        # outputs = [batch_size, 1, output_size = 20]

        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden

    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]

        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features, encoder_hidden)

        s = encoder_hidden[0][-1, :, :].unsqueeze(0)

        #decoder_input = encoder_output[:,-1,:].unsqueeze(1)
        decoder_input = torch.tanh(self.decoder_fc_0(lstm_input_features[:, -1, :].unsqueeze(1)))


        decoder_output, decoder_hidden = self.LSTM_decoder(decoder_input, s, encoder_output)  # [batch_size, 1, output_size]

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_attention_decoder_images_PR_many (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_attention_decoder_images_PR_many, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

        self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_00 = nn.Linear(encoder_input_size, decoder_hidden_size)  # for the decoder_input #10260 calculated
        self.decoder_fc_0 = nn.Linear(output_size, decoder_hidden_size)  # for the decoder_input #10260 calculated
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size + encoder_hidden_size, int(decoder_hidden_size / 2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size / 2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]

        s = s[0].repeat(seq_len, 1, 1) #in LSTM
        #s = s.repeat(seq_len, 1, 1)  # in GRU

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0, 1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)

    def LSTM_decoder(self, decoder_input, s, encoder_output):
        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]
        a = self.attention_net(s, encoder_output).transpose(1, 2)

        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)

        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]

        decoder_input = torch.cat((decoder_input, c), dim=2)

        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        decoder_hidden = s

        decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)

        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c), dim=2)



        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))

        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        #print(len(features))
        #print(len(features[0][0]))
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)


        #print(lstm_input_features.shape)

        #print(lstm_input_features.shape)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)

        if self.cuda_p:
            output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
        else:
            output = torch.zeros(pr_s[0].size(0), 1, 2)

        decoder_input = torch.tanh(self.decoder_fc_00(lstm_input_features[:, -1, :].unsqueeze(1)))

        #s = encoder_hidden[-1, :, :].unsqueeze(0) # for GRU
        s = encoder_hidden #for LSTM
        for t in range(0, predict_n_pr):
            # decoder_output = [batch_size, 1, output_size = 2]

            decoder_output, s = self.LSTM_decoder(decoder_input, s, encoder_output)

            output = torch.cat([output, decoder_output], 1)

            top1 = decoder_output.argmax(1).float().unsqueeze(1)

            decoder_input = torch.tanh(self.decoder_fc_0(top1))


        # output = [batchize, predict_n_pr +1 , output_size = 2]
        output = output[:, 1:, :]
        decoder_hidden = s
        # output = [batch_size, predict_n_pr, output_size = 2]
        return output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_GRU_attention_decoder_images_PR_many (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_GRU_attention_decoder_images_PR_many, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

        self.decoder_gru = nn.GRU(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_00 = nn.Linear(encoder_input_size, decoder_hidden_size)  # for the decoder_input #10260 calculated
        self.decoder_fc_0 = nn.Linear(output_size, decoder_hidden_size)  # for the decoder_input #10260 calculated
        self.decoder_fc_1 = nn.Linear(decoder_hidden_size + encoder_hidden_size, int(decoder_hidden_size / 2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size / 2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]

        s = s[0].repeat(seq_len, 1, 1) #in LSTM
        #s = s.repeat(seq_len, 1, 1)  # in GRU

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0, 1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)

    def GRU_decoder(self, decoder_input, s, encoder_output):
        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]
        a = self.attention_net(s, encoder_output).transpose(1, 2)

        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)

        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]

        decoder_input = torch.cat((decoder_input, c), dim=2)

        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        decoder_hidden = s

        decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)

        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c), dim=2)



        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))

        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        #print(len(features))
        #print(len(features[0][0]))
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)


        #print(lstm_input_features.shape)

        #print(lstm_input_features.shape)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)

        if self.cuda_p:
            output = torch.zeros(pr_s[0].size(0), 1, 2).cuda()
        else:
            output = torch.zeros(pr_s[0].size(0), 1, 2)

        decoder_input = torch.tanh(self.decoder_fc_00(lstm_input_features[:, -1, :].unsqueeze(1)))

        #s = encoder_hidden[-1, :, :].unsqueeze(0) # for GRU
        s = encoder_hidden[0] #for LSTM
        for t in range(0, predict_n_pr):
            # decoder_output = [batch_size, 1, output_size = 2]

            decoder_output, s = self.GRU_decoder(decoder_input, s, encoder_output)

            output = torch.cat([output, decoder_output], 1)

            top1 = decoder_output.argmax(1).float().unsqueeze(1)

            decoder_input = torch.tanh(self.decoder_fc_0(top1))


        # output = [batchize, predict_n_pr +1 , output_size = 2]
        output = output[:, 1:, :]
        decoder_hidden = s
        # output = [batch_size, predict_n_pr, output_size = 2]
        return output, encoder_hidden, decoder_hidden

class CNN_GRU_encoder_decoder_images_PR (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
        super(CNN_GRU_encoder_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_gru = nn.GRU(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_gru = nn.GRU(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def GRU_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_gru(inputs, hiddens)
        return outputs, hiddens


    def GRU_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_gru(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        gru_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        gru_input_features = torch.cat(gru_input_features, 2).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.GRU_encoder(gru_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.GRU_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


class CNN_GRU_encoder_attention_decoder_images_PR (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 600, decoder_hidden_size = 150, output_size = 20):
        super(CNN_GRU_encoder_attention_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_gru = nn.GRU(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

        self.decoder_gru = nn.GRU(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_0 = nn.Linear(encoder_input_size, decoder_hidden_size)  # for the decoder_input
        self.decoder_fc_1 = nn.Linear(encoder_hidden_size*2 + decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def GRU_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_gru(inputs, hiddens)
        return outputs, hiddens

    def attention_net(self, s, encoder_output):
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_len, encoder_hidden_size]

        batch_size = encoder_output.shape[0]
        seq_len = encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # s = [seq_len, batch_size, dedcoder_hidden_size]
        # encoder_output = [batch_size, seq_length, encoder_hidden_size]
        s = s.repeat(seq_len, 1, 1)

        # energy = [batch_size, seq_len, decoder_hiden_size]
        energy = torch.tanh(self.attn(torch.cat((s.transpose(0, 1), encoder_output), dim=2)))

        # attention = [batch_size, seq_len, 1]
        attention = self.v(energy)

        return F.softmax(attention, dim=1)

    def GRU_decoder(self, decoder_input, s, encoder_output):
        # decoder_input = [batch_size, 1, output_size = 20]
        # s = [1, batch_size, decoder_hidden_size]
        # encoder_output = [batch_size, seq_length = 1, encoder_hidden_size]

        # a = [batch_size, 1, seq_len]
        a = self.attention_net(s, encoder_output).transpose(1, 2)

        # c = [batch_size, 1, encoder_hidden_size]
        c = torch.bmm(a, encoder_output)

        # decoder_input = [batch_size, selen = 1, encoder_hidden_size * 2]

        decoder_input = torch.cat((decoder_input, c), dim=2)

        # decoder_output = [batch_size, seq_len = 1, decoder_hidden_size]
        # decoder_hidden = [n_layers * num_directions = 1, batch_size, decoder_hidden_size]

        # decoder_hidden = [s, c.transpose(0,1)] in LSTM

        decoder_hidden = s # in GRU


        decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)

        # encoder_output = [batch_size, 1, encoder_hidden_size]
        # dec_output = [batch_size, 1, decoder_hidden_size]
        # c = [batch_size, 1, encoder_hidden_size]

        # outputs = [batch_size, 1, encoder_hidden_size * 2 + decoder_hidden_size]
        outputs = torch.cat((decoder_output, c, encoder_output), dim=2)

        # outputs = [batch_size, 1, output_size = 20]
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, decoder_hidden


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        gru_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        gru_input_features = torch.cat(gru_input_features, 2).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.GRU_encoder(gru_input_features,  encoder_hidden)
        #s = encoder_hidden[0][-1,:,:].unsqueeze(0) # in LSTM
        s = encoder_hidden[-1,:,:].unsqueeze(0)

        decoder_input = torch.tanh(self.decoder_fc_0(gru_input_features[:, -1, :].unsqueeze(1)))

        decoder_output, decoder_hidden = self.GRU_decoder(decoder_input, s, encoder_output) #[batch_size, 1, output_size]

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1) #[batch_size,predict_n_pr, 2]

        return decoder_output, encoder_hidden, decoder_hidden



# the codes below are for transformer:
#
# ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
# emsize = 200 # embedding dimension
# nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2 # the number of heads in the multiheadattention models
# dropout = 0.2 # the dropout value
# model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(TransformerModel_PR, self).__init__()
        self.cuda_p = cuda
        #transformer parameters
        self.ntoken = 2 #pitch_and_roll
        self.ninp = self.ntoken #embeded_dim
        self.nhead = 1
        self.nlayers = 1
        self.dropout = 0.4
        self.src_mask = None
        self.tgt_masak = None
        self.nhid = encoder_hidden_size

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.pos_encoder = PositionalEncoding(self.ninp, self.dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)

        decoder_layers = nn.TransformerDecoderLayer(self.ninp, self.nhead, self.nhid, self.dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, self.nlayers)

    #the two init functions have no sense but to maintain the structure
    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr):


        PR_src  = [pr_s[i] for i in range(int(use_n_im))]
        src = torch.cat(PR_src, 1).view(pr_s[0].size(0), int(use_n_im), -1)


  #      PR_tgt  = [pr_s[i] for i in range(int(use_n_im /2)+1, use_n_im)]
  #      tgt = torch.cat(PR_tgt, 1).view(pr_s[0].size(0), use_n_im - int(use_n_im/2), -1)


        src = self.pos_encoder(src * math.sqrt(self.ninp))
  #      tgt = self.pos_encoder(tgt * math.sqrt(self.ninp))

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
        #     device = tgt.device
        #     mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
        #     self.src_mask = mask

        src = F.relu(src)
        tgt = src #F.relu(tgt)

        memory = self.transformer_encoder(src, self.src_mask)

        decoder_output = self.transformer_decoder(tgt, memory)

        return decoder_output

