import torch.nn as nn

class Hand2Object_unet(nn.Module):
    def __init__(self):
        embed_size = 256
        super().__init__()
        self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(in_channels = 138,out_channels= embed_size, kernel_size = 3, padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
			nn.MaxPool1d(kernel_size=2, stride=2)
		)

		# Adopted the same UNet structure as body2hand
        self.conv5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size)
		)
        self.conv6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size)
		)

        self.conv7 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

        self.conv8 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

        self.conv9 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

        self.conv10 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

        self.skip1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
        self.skip2 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
        self.skip4 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
        self.skip5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

        self.decoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),

			nn.Dropout(0.5),
			nn.ConvTranspose1d(embed_size, 18, 7, stride=2, padding=3, output_padding=1),
			nn.ReLU(True),
			nn.BatchNorm1d(18),

			nn.Dropout(0.5),
			nn.Conv1d(18, 18, 7, padding=3),
		)
        

    def upsample(self, tensor, shape):
        return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]]

    def forward(self, input_):
        input_ = input_.permute(0, 2, 1)
        fourth_block = self.encoder(input_)
        
        fifth_block = self.conv5(fourth_block)
        sixth_block = self.conv6(fifth_block)
        seventh_block = self.conv7(sixth_block)
        eighth_block = self.conv8(seventh_block)
        ninth_block = self.conv9(eighth_block)
        tenth_block = self.conv10(ninth_block)
        
        ninth_block = tenth_block + ninth_block
        ninth_block = self.skip1(ninth_block)
        
        eighth_block = ninth_block + eighth_block
        eighth_block = self.skip2(eighth_block)
        
        sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
        sixth_block = self.skip4(sixth_block)
        
        fifth_block = sixth_block + fifth_block		
        fifth_block = self.skip5(fifth_block)
        
        output = self.decoder(fifth_block)
        output = output.permute(0, 2, 1)
        return output 