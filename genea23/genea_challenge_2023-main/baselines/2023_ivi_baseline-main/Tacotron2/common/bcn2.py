from collections import OrderedDict

import torch
from torch import nn
import math


def positional_encoding(shape, max_len, absolut_pos):
    pos_enc = []
    positions = torch.FloatTensor([[i + pos for i in range(max_len)] for pos in absolut_pos])
    for pos in positions:
        pos = torch.unsqueeze(pos, 0)
        div_term = torch.unsqueeze(torch.exp(torch.arange(0, shape[1], 2) * (-math.log(10000.0) / shape[1])), 1)
        pe = torch.zeros(1, shape[1], max_len)
        rounding_error = 2 * div_term.shape[0] - shape[1]
        pe[0, 0::2, :] = torch.sin(pos * div_term)
        pe[0, 1::2, :] = torch.cos(pos * div_term[:div_term.shape[0] - rounding_error])
        pos_enc.append(pe)
    torch.cat(pos_enc, dim=0).cuda()
    return torch.cat(pos_enc, dim=0).cuda()


class Discriminator(nn.Module):
    def __init__(self, action_dim, input_length):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            LinearLayer(action_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            LinearLayer(input_length * 8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            LinearLayer(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            LinearLayer(256, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class AttentionHead(nn.Module):
    def __init__(self, params):
        super(AttentionHead, self).__init__()
        self.conv_embed = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(params.conv_in, params.embed_dim, params.kernel_size, bias=False)),
                ('Conv2', nn.Conv1d(params.embed_dim, params.embed_dim, params.kernel_size, bias=False)),
                ('Conv3', nn.Conv1d(params.embed_dim, params.embed_dim, params.kernel_size, bias=False)),
                ('Norm', nn.BatchNorm1d(params.embed_dim))
            ])
        )
        self.attention_head = nn.MultiheadAttention(params.embed_dim, params.num_heads, params.dropout, params.bias,
                                                    batch_first=True)
        self.feedforward = FeedforwardNetwork(params.feedforward)

    def forward(self, query, key, value, mask=None):
        key = self.conv_embed(key).transpose(1, 2)
        value = self.conv_embed(value).transpose(1, 2)
        query = self.conv_embed(query).transpose(1, 2)
        if mask is not None:
            output_mha, output_weights = self.attention_head(query, key, value, attn_mask=mask)
        else:
            output_mha, output_weights = self.attention_head(query, key, value)
        output_mha = nn.functional.normalize(torch.add(output_mha, value)).transpose(1, 2)
        output = self.feedforward(output_mha.transpose(1, 2))
        output = nn.functional.normalize(torch.add(output_mha.transpose(1, 2), output))
        return output


class FeedforwardNetwork(nn.Module):
    def __init__(self, params):
        super(FeedforwardNetwork, self).__init__()
        self.linear1 = LinearLayer(params.in_dim, params.lin_dim, False)
        self.drop_out = nn.Dropout(0.3)
        self.silu = nn.SiLU()
        self.linear2 = LinearLayer(params.in_dim, params.lin_dim, False)
        self.drop_out1 = nn.Dropout(0.3)
        self.output_layer = LinearLayer(params.lin_dim, params.out_dim, False)

    def forward(self, x):
        x = self.drop_out(x)
        x1 = self.linear1(x)
        x1 = self.silu(x1)
        x2 = self.linear2(x)
        x3 = x2 * x1
        x3 = self.drop_out1(x3)
        output = self.output_layer(x3)
        return output


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear(x)


class TAE(nn.Module):
    def __init__(self, params):
        super(TAE, self).__init__()
        self.text_audio_attention = AttentionHead(params.text_audio_attention)
        self.conv_layers = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(params.conv_in, params.conv_dim1, params.kernel_size)),
                ('Conv2', nn.Conv1d(params.conv_dim1, params.conv_dim, params.kernel_size)),
                ('Conv3', nn.Conv1d(params.conv_dim, params.conv_dim, params.kernel_size)),
                ('Norm', nn.BatchNorm1d(params.conv_dim))
            ])
        )

    def forward(self, text_audio, max_len, absolut_pos):
        text_audio += positional_encoding(text_audio.shape, max_len[0], absolut_pos)
        text_audio_attention = self.text_audio_attention(text_audio, text_audio, text_audio)
        text_audio_embedding = self.conv_layers(text_audio_attention.transpose(1, 2))

        return text_audio_embedding, text_audio_attention


class ControlNetwork(nn.Module):
    def __init__(self, params):
        super(ControlNetwork, self).__init__()
        self.control_conv1 = nn.Conv1d(params.control_conv, params.control_dim,
                                       kernel_size=1)
        self.control_conv2 = nn.Conv1d(params.control_dim, params.lin_dim,
                                       kernel_size=1)
        self.control_conv3 = nn.Conv1d(params.lin_dim, params.lin_dim,
                                       kernel_size=1)
        self.control_norm = nn.BatchNorm1d(params.lin_dim)
        self.control_network = nn.ModuleList()
        for joints in params.controllable_joints:
            self.control_network.append(nn.Sequential(
                OrderedDict([
                    ('Embedding_input', LinearLayer(params.lin_dim, params.lin_out)),
                    ('ReLu', nn.ReLU()),
                    ('Output', LinearLayer(params.lin_out, joints))
                ])
            ))

    def forward(self, output_combined, gamma, beta):
        gamma = gamma.transpose(1, 2)
        beta = beta.transpose(1, 2)
        input_main = self.control_conv1(output_combined)
        input_main *= gamma
        input_main = self.control_conv2(input_main)
        input_main += beta
        input_main = self.control_conv3(input_main)
        input_main = self.control_norm(input_main)
        input_main = input_main.transpose(1, 2)
        output = []
        for m in self.control_network:
            output.append(m(input_main))
        return torch.cat(output, dim=-1)


class FEIN(nn.Module):
    def __init__(self, params):
        super(FEIN, self).__init__()
        self.gesture_attention = AttentionHead(params.gesture_attention)
        self.convolutions_sequence = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(params.conv_in, params.conv_dim1, params.kernel_size_in, padding=1)),
                ('Conv2', nn.Conv1d(params.conv_dim1, params.conv_dim1, params.kernel_size)),
                ('Conv3', nn.Conv1d(params.conv_dim1, params.conv_dim1, params.kernel_size)),
                ('BatchNorm', nn.BatchNorm1d(params.conv_dim1))
            ])
        )
        self.convolutions = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(params.conv_dim2, params.conv_dim3, params.kernel_size)),
                ('Conv2', nn.Conv1d(params.conv_dim3, params.conv_dim4, params.kernel_size)),
                ('Conv3', nn.Conv1d(params.conv_dim4, params.conv_dim4, params.kernel_size)),
                ('BatchNorm', nn.BatchNorm1d(params.conv_dim4))
            ])
        )
        self.gamma_out = nn.Sequential(
            OrderedDict([
                ('Linear1', LinearLayer(params.in_dim, params.lin_dim, bias=False)),
                ('SiLU', nn.SiLU()),
                ('Linear2', LinearLayer(params.lin_dim, params.gamnma_out, bias=False)),
                ('SiLU', nn.SiLU()),
                ('Norm', nn.LayerNorm(params.gamnma_out))
            ])
        )
        self.beta_out = nn.Sequential(
            OrderedDict([
                ('Linear1', LinearLayer(params.in_dim, params.lin_dim, bias=False)),
                ('SiLU', nn.SiLU()),
                ('Linear2', LinearLayer(params.lin_dim, params.beta_out, bias=False)),
                ('SiLU', nn.SiLU()),
                ('Norm', nn.LayerNorm(params.beta_out))
            ])
        )

    def forward(self, audio_text_attention):
        sequence_prediction = self.convolutions_sequence(audio_text_attention)
        x = self.convolutions(sequence_prediction.transpose(1, 2))
        x = self.gesture_attention(x, x, x)
        gamma = self.gamma_out(x)
        beta = self.beta_out(x)
        return gamma, beta


class BCN2(nn.Module):
    def __init__(self, params):
        super(BCN2, self).__init__()
        self.tae = TAE(params.tae)
        self.fein = FEIN(params.fein)
        self.beta_attention = nn.MultiheadAttention(512, 8, 0.5)
        self.beta_ffn = nn.Sequential(
            OrderedDict([
                ('FFN', FeedforwardNetwork(params.beta_attention)),
                ('Norm', nn.LayerNorm(128))
            ]))
        self.gamma_attention = nn.MultiheadAttention(512, 8, 0.5, batch_first=True)
        self.gamma_ffn = nn.Sequential(
            OrderedDict([
                ('FFN', FeedforwardNetwork(params.gamma_attention)),
                ('Norm', nn.LayerNorm(256))
            ]))
        self.control_network = ControlNetwork(params.control_network)


    def forward(self, x):
        audio, text, output_length, absolut_pos, pre_gestures = x
        text_audio = torch.cat([audio, text], dim=1)
        text_audio_attention, text_audio_embedding = self.tae(text_audio, output_length, absolut_pos)
        gamma, beta = self.fein(pre_gestures.transpose(1, 2))
        gamma_attention, _ = self.gamma_attention(gamma, text_audio_embedding, gamma)
        gamma_attention = self.gamma_ffn(gamma_attention)
        beta_attention, _ = self.beta_attention(beta, text_audio_embedding, beta)
        beta_attention = self.beta_ffn(beta_attention)
        return self.control_network(text_audio_attention, gamma_attention, beta_attention)

    def inference(self, x, step_size=50, window_size=200):
        audio, text, output_length, absolut_pos, pre_gestures = x
        padding = window_size - (audio.shape[2] % (window_size - step_size))
        text_audio = torch.cat([audio, text], dim=1)
        text_audio = torch.cat([text_audio, torch.zeros((text_audio.shape[0], text_audio.shape[1], padding)).cuda()],
                               dim=-1)
        iterations = int(text_audio.shape[2] / step_size)
        motion_out = []
        for i in range(iterations - 3):
            motion_out.append(pre_gestures)
            text_audio_in = text_audio[:, :, step_size * i: window_size + (step_size * i)]
            text_audio_attention, text_audio_embedding = self.tae(text_audio_in, [window_size], absolut_pos)
            gamma, beta = self.fein(pre_gestures.transpose(1, 2))
            gamma_attention, _ = self.gamma_attention(gamma, text_audio_embedding, gamma)
            gamma_attention = self.gamma_ffn(gamma_attention)
            beta_attention, _ = self.beta_attention(beta, text_audio_embedding, beta)
            beta_attention = self.beta_ffn(beta_attention)
            motion = self.control_network(text_audio_attention, gamma_attention, beta_attention)
            absolut_pos += step_size
            pre_gestures = motion.transpose(1, 2)[:, :, window_size - 50:]
        motion_out.append(motion.transpose(1, 2)[:, :, :window_size - padding])

        return torch.cat(motion_out, dim=2)
