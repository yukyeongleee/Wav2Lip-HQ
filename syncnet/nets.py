import torch
import torch.nn as nn
import torch.nn.functional as F

from wav2lip.blocks import conv_block

class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()

        self.audio_encoder = AudioEncoder()
        self.face_encoder = FaceEncoder()

    def forward(self, audio_sequences, face_sequences):
        audio_embedding = self.audio_encoder(audio_sequences)
        face_embedding = self.face_encoder(face_sequences)
        
        return audio_embedding, face_embedding


"""
SyncNet submodules
"""

class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()

        self.encoder = nn.Sequential(
            conv_block(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            conv_block(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(128, 256, kernel_size=3, stride=2, padding=1),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(256, 512, kernel_size=3, stride=2, padding=1),
            conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(512, 512, kernel_size=3, stride=2, padding=1),
            conv_block(512, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, face_sequences):
        face_embedding = self.encoder(face_sequences)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return face_embedding


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

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
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(256, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, audio_sequences):
        audio_embedding = self.encoder(audio_sequences)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        
        return audio_embedding