# from collections import OrderedDict

# import torch
# from torch import nn
# import math


# #################### Model 2 (Transformer like) ########################

# class AttentionHead2(nn.Module):
#     def __init__(self, params):
#         super(AttentionHead2, self).__init__()
#         self.conv_embed = nn.Conv1d(params.conv_in, params.embed_dim, params.kernel_size, bias=False)
#         self.attention_head = nn.MultiheadAttention(params.embed_dim, params.num_heads, params.dropout, params.bias,
#                                                     batch_first=True)
#         self.feedforward = FeedforwardNetwork(params.feedforward)

#     def forward(self, query, key, value, mask=None):
#         key = self.conv_embed(key).transpose(1, 2)
#         value = self.conv_embed(value).transpose(1, 2)
#         query = self.conv_embed(query).transpose(1, 2)
#         if mask is not None:
#             output_mha, output_weights = self.attention_head(query, key, value, attn_mask=mask)
#         else:
#             output_mha, output_weights = self.attention_head(query, key, value)
#         output_mha = nn.functional.normalize(torch.add(output_mha, value)).transpose(1, 2)
#         output = self.feedforward(output_mha.transpose(1, 2))
#         output = nn.functional.normalize(torch.add(output_mha.transpose(1, 2), output))
#         return output


# class FeedforwardNetwork(nn.Module):
#     def __init__(self, params):
#         super(FeedforwardNetwork, self).__init__()
#         self.linear1 = LinearLayer(params.in_dim, params.lin_dim, False)
#         self.drop_out = nn.Dropout(0.3)
#         self.silu = nn.SiLU()
#         self.linear2 = LinearLayer(params.in_dim, params.lin_dim, False)
#         self.drop_out1 = nn.Dropout(0.3)
#         self.output_layer = LinearLayer(params.lin_dim, params.out_dim, False)

#     def forward(self, x):
#         x = self.drop_out(x)
#         x1 = self.linear1(x)
#         x1 = self.silu(x1)
#         x2 = self.linear2(x)
#         x3 = x2 * x1
#         x3 = self.drop_out1(x3)
#         output = self.output_layer(x3)
#         return output


# class LinearLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
#         super(LinearLayer, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim, bias)
#         torch.nn.init.xavier_uniform_(self.linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

#     def forward(self, x):
#         return self.linear(x)


# class ControlNetwork(nn.Module):
#     def __init__(self, params):
#         super(ControlNetwork, self).__init__()
#         self.control_conv1 = nn.Conv1d(params.control_in, params.control_dim,
#                                        kernel_size=1)
#         self.control_conv2 = nn.Conv1d(params.control_dim, params.control_dim,
#                                        kernel_size=1)
#         self.control_conv3 = nn.Conv1d(params.control_dim, params.control_dim,
#                                        kernel_size=1)
#         self.control_mean = nn.BatchNorm1d(params.control_in)
#         self.control_network = nn.ModuleList()
#         for joints in params.controllable_joints:
#             self.control_network.append(nn.Sequential(
#                 OrderedDict([
#                     ('Embedding_input', LinearLayer(params.control_dim, params.control_out)),
#                     ('SiLU', nn.SiLU()),
#                     ('Output', LinearLayer(params.control_out, joints))
#                 ])
#             ))

#     def forward(self, x, gamma, beta):
#         outputs_main = []
#         gamma = gamma.transpose(1, 2)
#         beta = beta.transpose(1, 2)
#         input_main = self.control_conv1(x.transpose(1, 2))
#         input_main *= gamma
#         input_main = self.control_conv2(input_main)
#         input_main += beta
#         input_main = self.control_conv3(input_main)
#         input_main = self.control_mean(input_main)
#         input_main = input_main.transpose(1, 2)
#         for m in self.control_network:
#             outputs_main.append(m(input_main))
#         return torch.cat(outputs_main, dim=-1)


# class FEIN(nn.Module):
#     def __init__(self, params):
#         super(FEIN, self).__init__()
#         self.convolutions_input = nn.Sequential(
#             OrderedDict([
#                 ('Conv1', nn.Conv1d(params.conv_in, params.conv_dim, params.kernel_size, bias=False)),
#                 ('Conv2', nn.Conv1d(params.conv_dim, params.conv_dim, params.kernel_size, bias=False)),
#                 ('Conv3', nn.Conv1d(params.conv_dim, params.conv_out, params.kernel_size, bias=False)),
#                 ('BatchNorm', nn.BatchNorm1d(params.conv_out))
#             ])
#         )
#         self.convolutions_gesture = nn.Sequential(
#             OrderedDict([
#                 ('Conv1', nn.Conv1d(params.gesture_in, params.conv_dim, params.kernel_size, bias=False)),
#                 ('Conv2', nn.Conv1d(params.conv_dim, params.conv_dim, params.kernel_size, bias=False)),
#                 ('Conv3', nn.Conv1d(params.conv_dim, params.conv_out, params.kernel_size, bias=False)),
#                 ('BatchNorm', nn.BatchNorm1d(params.conv_out))
#             ])
#         )
#         self.attention_gesture = AttentionHead2(params.gesture_attention)
#         self.gamma_out = nn.Sequential(
#             OrderedDict([
#                 ('Linear1', LinearLayer(params.conv_out, params.lin_dim)),
#                 ('SiLU', nn.SiLU()),
#                 ('Linear2', LinearLayer(params.lin_dim, params.lin_dim)),
#                 ('SiLU', nn.SiLU()),
#                 ('Norm', nn.LayerNorm(params.lin_dim))
#             ])
#         )
#         self.beta_out = nn.Sequential(
#             OrderedDict([
#                 ('Linear1', LinearLayer(params.conv_out, params.lin_dim)),
#                 ('SiLU', nn.SiLU()),
#                 ('Linear2', LinearLayer(params.lin_dim, params.lin_dim)),
#                 ('SiLU', nn.SiLU()),
#                 ('Norm', nn.LayerNorm(params.lin_dim))
#             ])
#         )

#     def forward(self, x, pre_gesture):
#         x = self.convolutions_input(x)
#         pre_gesture = self.convolutions_gesture(pre_gesture)
#         x = self.attention_gesture(pre_gesture, x, pre_gesture)  # .transpose(1, 2)
#         gamma = self.gamma_out(x)
#         beta = self.beta_out(x)
#         return gamma, beta


# class BGTransformer(nn.Module):
#     def __init__(self, params):
#         super(BGTransformer, self).__init__()
#         self.text_attention = AttentionHead2(params.text_attention)
#         self.audio_attention = AttentionHead2(params.audio_attention)
#         self.combined_attention_conv = nn.Sequential(
#             OrderedDict([
#                 ('Conv',
#                  nn.Conv1d(params.combined_attention_lin_dim, params.combined_attention_embed_dim, kernel_size=1,
#                            bias=False)),
#                 ('Norm', nn.BatchNorm1d(params.combined_attention_embed_dim))
#             ])
#         )

#         self.combined_attention_main = AttentionHead2(params.combined_attention)
#         self.output_layers = nn.Sequential(
#             OrderedDict([
#                 ('Linear', nn.Linear(params.combined_attention_embed_dim, params.combined_attention_embed_dim)),
#                 ('SiLU', nn.SiLU()),
#                 ('Norm', nn.LayerNorm(params.combined_attention_embed_dim))
#             ])
#         )

#     def forward(self, x):
#         audio_in, text_in = x

#         audio_tensor = torch.nested.nested_tensor([audio_in, audio_in, audio_in]).cuda()
#         text_tensor = torch.nested.nested_tensor([text_in, text_in, text_in]).cuda()
#         audio_attention = self.audio_attention(audio_tensor)
#         text_attention = self.text_attention(text_tensor)

#         audio_text_attention = torch.cat([audio_attention, text_attention], dim=2)
#         audio_text_attention = self.combined_attention_conv(audio_text_attention)

#         audio_text_tensor = torch.nested.nested_tensor(
#             [audio_text_attention, audio_text_attention, audio_text_attention])
#         out_mha = self.combined_attention_main(audio_text_tensor)
#         out_mha = nn.functional.normalize(torch.add(out_mha, audio_text_attention))
#         out_ff = self.combined_feedforward(out_mha)
#         out = self.output_layers(out_ff)
#         return out


# class BCNetwork(nn.Module):
#     def __init__(self, params):
#         super(BCNetwork, self).__init__()
#         self.bg_transformer = BGTransformer(params)
#         self.fein_layer_main = FEIN(params.fein)
#         self.control_network = ControlNetwork(params.control_network)

#     def forward(self, x):
#         audio_in, text_in, audio_in_inter, text_in_inter, max_len, absolut_pos, gesture, gesture_inter = x

#         audio_in += positional_encoding(audio_in.shape, max_len[0], absolut_pos)
#         text_in += positional_encoding(text_in.shape, max_len[0], absolut_pos)
#         audio_in_inter += positional_encoding(audio_in_inter.shape, max_len[0], absolut_pos)
#         text_in_inter += positional_encoding(text_in_inter.shape, max_len[0], absolut_pos)

#         gesture += positional_encoding(gesture.shape, max_len[0], absolut_pos - 50)
#         gesture_inter += positional_encoding(gesture_inter.shape, max_len[0], absolut_pos - 100)
#         input_x = torch.cat([audio_in, audio_in_inter], dim=1), torch.cat([text_in, text_in_inter], dim=1)
#         gamma, beta = self.fein_layer_main(torch.cat([input_x[0], input_x[1]], dim=1),
#                                            torch.cat([gesture, gesture_inter], dim=1))
#         output_combined = self.bg_transformer(input_x)

#         return self.control_network(output_combined, gamma, beta)


# class Discriminator(nn.Module):
#     def __init__(self, action_dim, input_length):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(action_dim, 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Flatten(),
#             nn.Linear(input_length * 8, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#         )

#     def forward(self, img):
#         validity = self.model(img)
#         return validity


# def positional_encoding_inference(shape, max_len):
#     pos = torch.FloatTensor([i for i in range(max_len)])
#     div_term = torch.unsqueeze(torch.exp(torch.arange(0, shape[0], 2) * (-math.log(10000.0) / shape[0])), 1)
#     pe = torch.zeros(shape)
#     rounding_error = 2 * div_term.shape[0] - shape[0]
#     pe[0::2, :] = torch.sin(pos * div_term)
#     pe[1::2, :] = torch.cos(pos * div_term[:div_term.shape[0] - rounding_error])
#     return pe


# def positional_encoding(shape, max_len, absolut_pos):
#     pos_enc = []
#     positions = torch.FloatTensor([[i + pos for i in range(max_len)] for pos in absolut_pos])
#     for pos in positions:
#         pos = torch.unsqueeze(pos, 0)
#         div_term = torch.unsqueeze(torch.exp(torch.arange(0, shape[1], 2) * (-math.log(10000.0) / shape[1])), 1)
#         pe = torch.zeros(1, shape[1], max_len)
#         rounding_error = 2 * div_term.shape[0] - shape[1]
#         pe[0, 0::2, :] = torch.sin(pos * div_term)
#         pe[0, 1::2, :] = torch.cos(pos * div_term[:div_term.shape[0] - rounding_error])
#         pos_enc.append(pe)
#     torch.cat(pos_enc, dim=0).cuda()

#     return torch.cat(pos_enc, dim=0).cuda()

from collections import OrderedDict

import torch
from torch import nn
import math


#################### Model 2 (Transformer like) ########################

class AttentionHead2(nn.Module):
    def __init__(self, params):
        super(AttentionHead2, self).__init__()
        self.conv_embed = nn.Conv1d(params.conv_in, params.embed_dim, params.kernel_size, bias=False)
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


class ControlNetwork(nn.Module):
    def __init__(self, params):
        super(ControlNetwork, self).__init__()
        self.control_conv1 = nn.Conv1d(params.control_in, params.control_dim,
                                       kernel_size=1)
        self.control_conv2 = nn.Conv1d(params.control_dim, params.control_dim,
                                       kernel_size=1)
        self.control_conv3 = nn.Conv1d(params.control_dim, params.control_dim,
                                       kernel_size=1)
        self.control_mean = nn.BatchNorm1d(params.control_dim)
        self.control_network = nn.ModuleList()
        for joints in params.controllable_joints:
            self.control_network.append(nn.Sequential(
                OrderedDict([
                    ('Embedding_input', LinearLayer(params.control_dim, params.control_out)),
                    ('SiLU', nn.SiLU()),
                    ('Output', LinearLayer(params.control_out, joints))
                ])
            ))

    def forward(self, x, gamma, beta):
        outputs_main = []
        gamma = gamma.transpose(1, 2)
        beta = beta.transpose(1, 2)
        input_main = self.control_conv1(x.transpose(1, 2))
        # input_main *= gamma
        input_main = self.control_conv2(input_main)
        input_main += beta
        input_main = self.control_conv3(input_main)
        input_main = self.control_mean(input_main)
        input_main = input_main.transpose(1, 2)
        for m in self.control_network:
            outputs_main.append(m(input_main))
        return torch.cat(outputs_main, dim=-1)


class FEIN(nn.Module):
    def __init__(self, params):
        super(FEIN, self).__init__()
        self.convolutions_input = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(params.conv_in, params.conv_dim, params.kernel_size, bias=False)),
                ('Conv2', nn.Conv1d(params.conv_dim, params.conv_dim, params.kernel_size, bias=False)),
                ('Conv3', nn.Conv1d(params.conv_dim, params.conv_out, params.kernel_size, bias=False)),
                ('BatchNorm', nn.BatchNorm1d(params.conv_out))
            ])
        )
        self.convolutions_gesture = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(params.gesture_in, params.conv_dim, params.kernel_size, bias=False)),
                ('Conv2', nn.Conv1d(params.conv_dim, params.conv_dim, params.kernel_size, bias=False)),
                ('Conv3', nn.Conv1d(params.conv_dim, params.conv_out, params.kernel_size, bias=False)),
                ('BatchNorm', nn.BatchNorm1d(params.conv_out))
            ])
        )
        self.attention_gesture = AttentionHead2(params.gesture_attention)
        self.gamma_out = nn.Sequential(
            OrderedDict([
                ('Linear1', LinearLayer(params.conv_out, params.lin_dim)),
                ('SiLU', nn.SiLU()),
                ('Linear2', LinearLayer(params.lin_dim, params.lin_dim)),
                ('SiLU', nn.SiLU()),
                ('Norm', nn.LayerNorm(params.lin_dim))
            ])
        )
        self.beta_out = nn.Sequential(
            OrderedDict([
                ('Linear1', LinearLayer(params.conv_out, params.lin_dim)),
                ('SiLU', nn.SiLU()),
                ('Linear2', LinearLayer(params.lin_dim, params.lin_dim)),
                ('SiLU', nn.SiLU()),
                ('Norm', nn.LayerNorm(params.lin_dim))
            ])
        )

    def forward(self, x, pre_gesture):
        x = self.convolutions_input(x)
        pre_gesture = self.convolutions_gesture(pre_gesture)
        x = self.attention_gesture(pre_gesture, x, pre_gesture)  # .transpose(1, 2)
        gamma = self.gamma_out(x)
        beta = self.beta_out(x)
        return gamma, beta


class BGTransformer(nn.Module):
    def __init__(self, params):
        super(BGTransformer, self).__init__()
        self.text_attention = AttentionHead2(params.text_attention)
        self.audio_attention = AttentionHead2(params.audio_attention)
        self.combined_attention_conv = nn.Sequential(
            OrderedDict([
                ('Conv',
                 nn.Conv1d(params.combined_attention_lin_dim, params.combined_attention_embed_dim, kernel_size=1,
                           bias=False)),
                ('Norm', nn.BatchNorm1d(params.combined_attention_embed_dim))
            ])
        )

        self.combined_attention_main = AttentionHead2(params.combined_attention)
        self.output_layers = nn.Sequential(
            OrderedDict([
                ('Linear', nn.Linear(params.combined_attention_embed_dim, params.combined_attention_embed_dim)),
                ('SiLU', nn.SiLU()),
                ('Norm', nn.LayerNorm(params.combined_attention_embed_dim))
            ])
        )

    def forward(self, x):
        audio_in, text_in = x

        audio_attention = self.audio_attention(audio_in, audio_in, audio_in)
        text_attention = self.text_attention(text_in, text_in, text_in)

        audio_text_attention = torch.cat([audio_attention, text_attention], dim=2).transpose(1, 2)
        audio_text_attention = self.combined_attention_conv(audio_text_attention)

        out_mha = self.combined_attention_main(audio_text_attention, audio_text_attention, audio_text_attention)
        out = self.output_layers(out_mha)
        return out


class BCNetwork(nn.Module):
    def __init__(self, params):
        super(BCNetwork, self).__init__()
        self.bg_transformer = BGTransformer(params)
        self.fein_layer_main = FEIN(params.fein)
        self.control_network = ControlNetwork(params.control_network)

    def forward(self, x, warmup_period=100):
        audio_in, text_in, max_len, absolut_pos, gesture = x

        audio_in += positional_encoding(audio_in.shape, max_len[0], absolut_pos)
        text_in += positional_encoding(text_in.shape, max_len[0], absolut_pos)
        gesture += positional_encoding(gesture.shape, max_len[0], absolut_pos - warmup_period)
        input_x = audio_in, text_in

        gamma, beta = self.fein_layer_main(torch.cat([audio_in, text_in], dim=1), gesture)
        # gamma *= 0
        # beta *= 0
        output_combined = self.bg_transformer(input_x) #* 0

        return self.control_network(output_combined, gamma, beta).transpose(1, 2)

    def inference(self, x):
        audio_in, text_in, max_len, absolut_pos, gesture = x
        inter_gesture = gesture[:, 168:, :]

        x = (audio_in[:, :, 0:100], text_in[:, :, 0:100], [100], absolut_pos, gesture[:, :, :100])
        absolut_pos += 50
        pre_gesture = self.forward(x, 100)
        padding = 100 - (max_len % 100)
        audio_in = torch.cat([audio_in, torch.zeros(audio_in.shape[0], audio_in.shape[1], padding).cuda()], dim=2)
        text_in = torch.cat([text_in, torch.zeros(text_in.shape[0], text_in.shape[1], padding).cuda()], dim=2)
        inter_gesture = torch.cat(
            [inter_gesture, torch.zeros(inter_gesture.shape[0], inter_gesture.shape[1], padding).cuda()], dim=2)
        pre_gesture_in = torch.cat([pre_gesture, inter_gesture[:, :, 0:100]], dim=1)
        absolut_pos += 50
        motion_out = []
        #pre_gesture = torch.cat([gesture[:, :168, :50], pre_gesture], dim=2)
        l = 0
        for i in range(100, max_len + padding, 100):
            motion_out.append(pre_gesture)
            x = (audio_in[:, :, i:i + 100], text_in[:, :, i:i + 100], [100], absolut_pos, pre_gesture_in)
            pre_gesture = self.forward(x)
            absolut_pos += 100
            pre_gesture_in = torch.cat([pre_gesture, inter_gesture[:, :, i:i + 100]], dim=1)
        motion_out.append(pre_gesture[:, :, :100 - padding])
        return torch.cat(motion_out, dim=2)


class Discriminator(nn.Module):
    def __init__(self, action_dim, input_length):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(action_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(input_length * 8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


def positional_encoding_inference(shape, max_len):
    pos = torch.FloatTensor([i for i in range(max_len)])
    div_term = torch.unsqueeze(torch.exp(torch.arange(0, shape[0], 2) * (-math.log(10000.0) / shape[0])), 1)
    pe = torch.zeros(shape)
    rounding_error = 2 * div_term.shape[0] - shape[0]
    pe[0::2, :] = torch.sin(pos * div_term)
    pe[1::2, :] = torch.cos(pos * div_term[:div_term.shape[0] - rounding_error])
    return pe


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
