import torch
import torch.nn as nn

## Completed - TODO: Change the models to include text embeddings
## Completed - TODO: Add FC to reduce the text_embedding to the size of nt
class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, nte, nt):
        super(_netG, self).__init__()
        self.nt = nt
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nt, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            # Completed - TODO: check out paper's code and add layers if required

            ##there are more conv2d layers involved here in 
            # https://github.com/reedscot/icml2016/blob/master/main_cls.lua

            nn.Conv2d(ngf*8,ngf*2,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # nn.SELU(True),

            nn.Conv2d(ngf*2,ngf*2,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # nn.SELU(True),

            nn.Conv2d(ngf*2,ngf*8,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # nn.SELU(True),


            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),   
            nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            
            # Completed - TODO: check out paper's code and add layers if required
            
            ##there are more conv2d layers involved here in 
            # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
            
            
            nn.Conv2d(ngf*4,ngf,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # nn.SELU(True),

            nn.Conv2d(ngf,ngf,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # nn.SELU(True),

            nn.Conv2d(ngf,ngf*4,3,1,1),
            nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # nn.SELU(True),            
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # nn.SELU(True),
            
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # nn.SELU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.encode_text = nn.Sequential(
            nn.Linear(nte, nt), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input, text_embedding):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            encoded_text = nn.parallel.data_parallel(self.encode_text, text_embedding, )
            input_new = torch.cat((input, encoded_text))
            output = nn.parallel.data_parallel(self.main,input_new, range(self.ngpu))
        else:
            encoded_text = self.encode_text(text_embedding).view(-1,self.nt,1,1)
            output = self.main(torch.cat((input, encoded_text), 1))
        return output

## Completed - TODO: pass nt and text_embedding size to the G and D and add FC to reduce text_embedding_size to nt
class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf, nte, nt):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.nt = nt
        self.nte = nte
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),

            nn.Conv2d(ndf*8,ndf*2,1,1),
            # nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2,ndf*2,3,1,1),
            # nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2,ndf*8,3,1,1),
            # nn.Dropout2d(inplace=True),            
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf*8) x 4 x 4

        ## add another sequential plot after this line to add the embedding and process it to find a single ans
        # Completed - TODO: confirm if what we are doing is same as given in paper code
        self.encode_text = nn.Sequential(
            nn.Linear(nte, nt),
            nn.LeakyReLU(0.2, inplace=True)

        )

        self.concat_image_n_text = nn.Sequential(
            nn.Conv2d(ndf * 8 + nt, ndf * 8, 1, 1, 0, bias=False), ## TODO: Might want to change the kernel size and stride
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, text_embedding):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            encoded_img = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
            encoded_text = nn.parallel.data_parallel(self.encode_text, text_embedding, range(self.ngpu))
            ## add the same things as in the else part
        else:
            encoded_img = self.main(input)
            encoded_text = self.encode_text(text_embedding)
            encoded_text = encoded_text.view(-1, self.nt, 1,1)
            encoded_text = encoded_text.repeat(1, 1, 4, 4) ## can also directly expand, look into the syntax
            output = self.concat_image_n_text(torch.cat((encoded_img, encoded_text),1))

        return output.view(-1, 1).squeeze(1)
