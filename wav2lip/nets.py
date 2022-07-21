import torch
import torch.nn as nn

from wav2lip.blocks import conv_block, conv_block_no_norm, conv_trans_block

class LipGenerator(nn.Module):
    def __init__(self):
        super(LipGenerator, self).__init__()

        self.audio_encoder = AudioEncoderG()
        self.face_encoder = FaceEncoderG()
        self.face_decoder = FaceDecoderG()

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences [32, 16, 1, 80, 16] = [B, frames per segment, CH, W, H]
        # face_sequences [32, 6, 5, 96, 96] = [B, CH, frames per segment, W, H]

        B = audio_sequences.size(0)
        
        # Preprocess the input sequences
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0) # [512(=Bx16), 1, 80, 16]
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0) # [160(=Bx5), 6, 96, 96]

        # Encode
        audio_embedding = self.audio_encoder(audio_sequences) # [512(=Bx16), 512(=CH), 1, 1]
        _, face_embeddings = self.face_encoder(face_sequences)

        # Decode
        x = self.face_decoder(audio_embedding, face_embeddings)

        # Postprocess the network output
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
        
        return outputs

class LipDiscriminator(nn.Module):
    def __init__(self):
        super(LipDiscriminator, self).__init__()

        self.face_encoder = FaceEncoderD()
        self.binary_pred = nn.Sequential(
            nn.Conv2d(512, 1, 1, 1, 0), 
            nn.Sigmoid()
        )
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, generated_face_sequences):
        generated_face_sequences = self.to_2d(generated_face_sequences)
        generated_face_sequences = self.get_lower_half(generated_face_sequences)

        face_embedding = self.face_encoder(generated_face_sequences)
        # for f in self.face_encoder_blocks:
        #     false_feats = f(false_feats)

        # false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
        #                                 torch.ones((len(false_feats), 1)).cuda())

        return self.binary_pred(face_embedding).view(len(face_embedding), -1) # false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = self.face_encoder(face_sequences)

        return self.binary_pred(x).view(len(x), -1)


"""
Generator and Discriminator submodules
"""

class AudioEncoderG(nn.Module):
    def __init__(self):
        super(AudioEncoderG, self).__init__()

        self.encoder = nn.Sequential(
            conv_block(1, 32, kernel_size=3, stride=1, padding=1),
            conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(64, 128, kernel_size=3, stride=3, padding=1),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(256, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, audio_sequences):
        audio_embedding = self.encoder(audio_sequences)
        return audio_embedding


class FaceEncoderG(nn.Module):
    def __init__(self):
        super(FaceEncoderG, self).__init__()

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(conv_block(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(
                conv_block(16, 32, kernel_size=3, stride=2, padding=1),      # 48,48
                conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                conv_block(32, 64, kernel_size=3, stride=2, padding=1),      # 24,24
                conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                conv_block(64, 128, kernel_size=3, stride=2, padding=1),     # 12,12
                conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                conv_block(128, 256, kernel_size=3, stride=2, padding=1),    # 6,6
                conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                conv_block(256, 512, kernel_size=3, stride=2, padding=1),       # 3,3
                conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            
            nn.Sequential(
                conv_block(512, 512, kernel_size=3, stride=1, padding=0),       # 1, 1
                conv_block(512, 512, kernel_size=1, stride=1, padding=0)
            ),
        ])

    def forward(self, face_sequences):
        face_embeddings = []
        x = face_sequences
        for b in self.encoder_blocks:
            x = b(x)
            face_embeddings.append(x)

        return x, face_embeddings

class FaceEncoderD(nn.Module):
    def __init__(self):
        super(FaceEncoderD, self).__init__()

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(conv_block_no_norm(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(
                conv_block_no_norm(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
                conv_block_no_norm(64, 64, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                conv_block_no_norm(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
                conv_block_no_norm(128, 128, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                conv_block_no_norm(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
                conv_block_no_norm(256, 256, kernel_size=5, stride=1, padding=2)
            ),

            nn.Sequential(
                conv_block_no_norm(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
                conv_block_no_norm(512, 512, kernel_size=3, stride=1, padding=1)
            ),

            nn.Sequential(
                conv_block_no_norm(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
                conv_block_no_norm(512, 512, kernel_size=3, stride=1, padding=1)
            ),
            
            nn.Sequential(
                conv_block_no_norm(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
                conv_block_no_norm(512, 512, kernel_size=1, stride=1, padding=0)
            )
        ])

    def forward(self, face_sequences):
        x = face_sequences
        for b in self.encoder_blocks:
            x = b(x)
        
        return x

    # def forward(self, face_sequences):
    #     face_embeddings = []
    #     x = face_sequences
    #     for b in self.encoder_blocks:
    #         x = b(x)
    #         face_embeddings.append(x)

    #     return x, face_embeddings


class FaceDecoderG(nn.Module):
    def __init__(self):
        super(FaceDecoderG, self).__init__()

        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(conv_block(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(
                conv_trans_block(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
                conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ),

            nn.Sequential(
                conv_trans_block(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 6, 6

            nn.Sequential(
                conv_trans_block(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                conv_block(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 12, 12

            nn.Sequential(
                conv_trans_block(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 24, 24

            nn.Sequential(
                conv_trans_block(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
                conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 48, 48

            nn.Sequential(
                conv_trans_block(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 96, 96
        ]) 

        self.output_block = nn.Sequential(
            conv_block(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        ) 

    
    def forward(self, audio_embedding, face_embeddings):
        x = audio_embedding
        for b in self.decoder_blocks:
            x = b(x)
            try:
                x = torch.cat((x, face_embeddings[-1]), dim=1)
            except Exception as e: 
                print(e)
                print(f"audio: {x.size()} | face: {face_embeddings[-1].size()}")

            face_embeddings.pop()

        x = self.output_block(x)

        return x      
