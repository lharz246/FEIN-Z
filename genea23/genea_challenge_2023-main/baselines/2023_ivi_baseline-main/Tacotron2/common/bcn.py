from collections import OrderedDict

import torch
from torch import nn
import math
from .loss_bcgg import BCGLoss


#################### Model 2 (Transformer like) ########################

class AttentionHead2(nn.Module):
    def __init__(self, params):
        super(AttentionHead2, self).__init__()
        self.conv_embed = nn.Conv1d(params.conv_in, params.embed_dim, params.kernel_size)
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
    def __init__(self, params, dyadic=False):
        super(ControlNetwork, self).__init__()
        self.dyadic = dyadic
        self.control_conv1 = nn.Conv1d(params.control_in, params.control_in,
                                       kernel_size=1)
        self.control_conv2 = nn.Conv1d(params.control_in, params.control_in,
                                       kernel_size=1)
        self.control_conv3 = nn.Conv1d(params.control_in, params.control_in,
                                       kernel_size=1)
        self.control_mean = nn.BatchNorm1d(params.control_in)
        self.control_network = nn.ModuleList()
        for joints in params.controllable_joints:
            self.control_network.append(nn.Sequential(
                OrderedDict([
                    ('Embedding_input', LinearLayer(params.control_in, params.control_dim)),
                    ('ReLu', nn.ReLU()),
                    ('Output', LinearLayer(params.control_dim, joints))
                ])
            ))
        if self.dyadic:
            self.control_conv1_inter = nn.Conv1d(params.control_in, params.control_in,
                                                 kernel_size=1)
            self.control_conv2_inter = nn.Conv1d(params.control_in, params.control_in,
                                                 kernel_size=1)
            self.control_conv3_inter = nn.Conv1d(params.control_in, params.control_in,
                                                 kernel_size=1)
            self.control_mean_inter = nn.BatchNorm1d(params.control_in)
            self.control_network_inter = nn.ModuleList()
            for joints in params.controllable_joints:
                self.control_network_inter.append(nn.Sequential(
                    OrderedDict([
                        ('Embedding_input', LinearLayer(params.control_in, params.control_dim)),
                        ('ReLu', nn.ReLU()),
                        ('Output', LinearLayer(params.control_dim, joints))
                    ])
                ))

    def forward(self, output_combined, gamma_values, beta_values):
        if self.dyadic:
            gamma, gamma_inter = gamma_values
            beta, beta_inter = beta_values
            output_main = self.forward_control(output_combined, gamma_inter, beta_inter)
            output_inter = []
            gamma = gamma.transpose(1, 2)
            beta = beta.transpose(1, 2)
            input_inter = self.control_conv1_inter(output_combined.transpose(1, 2))
            input_inter *= gamma
            input_inter = self.control_conv2_inter(input_inter)
            input_inter += beta
            input_inter = self.control_conv3_inter(input_inter)
            input_inter = self.control_mean_inter(input_inter)
            input_inter = input_inter.transpose(1, 2)
            for m in self.control_network_inter:
                output_inter.append(m(input_inter))
            return output_main, torch.cat(output_inter, dim=-1)
        else:
            return self.forward_control(output_combined, gamma_values, beta_values)

    def forward_control(self, x, gamma, beta):
        outputs_main = []
        gamma = gamma.transpose(1, 2)
        beta = beta.transpose(1, 2)
        input_main = self.control_conv1(x.transpose(1, 2))
        input_main *= gamma
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
                ('Conv1', nn.Conv1d(params.conv_in, params.redu_dim, params.kernel_size)),
                ('Conv2', nn.Conv1d(params.redu_dim, params.redu_dim, params.kernel_size)),
                ('Conv3', nn.Conv1d(params.redu_dim, params.conv_dim, params.kernel_size)),
                ('BatchNorm', nn.BatchNorm1d(params.conv_dim))
            ])
        )
        self.convolutions_gesture = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(params.gesture_in, params.redu_dim, params.kernel_size)),
                ('Conv2', nn.Conv1d(params.redu_dim, params.redu_dim, params.kernel_size)),
                ('Conv3', nn.Conv1d(params.redu_dim, params.conv_dim, params.kernel_size)),
                ('BatchNorm', nn.BatchNorm1d(params.conv_dim))
            ])
        )
        self.attention_gesture = AttentionHead2(params.gesture_attention)
        self.gamma_out = nn.Sequential(
            OrderedDict([
                ('Linear1', LinearLayer(params.in_dim, params.lin_dim, False)),
                ('SiLU', nn.SiLU()),
                ('Linear2', LinearLayer(params.lin_dim, params.lin_out, False)),
                ('SiLU', nn.SiLU()),
                ('Norm', nn.LayerNorm(params.lin_out))
            ])
        )
        self.beta_out = nn.Sequential(
            OrderedDict([
                ('Linear1', LinearLayer(params.in_dim, params.lin_dim, False)),
                ('SiLU', nn.SiLU()),
                ('Linear2', LinearLayer(params.lin_dim, params.lin_out, False)),
                ('SiLU', nn.SiLU()),
                ('Norm', nn.LayerNorm(params.lin_out))
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
        self.combined_attention_lin = nn.Sequential(
            OrderedDict([
                ('Linear', nn.Linear(params.combined_attention_lin_dim, params.combined_attention_embed_dim)),
                ('Norm', nn.LayerNorm(params.combined_attention_embed_dim))
            ])
        )

        self.combined_attention_main = nn.MultiheadAttention(params.combined_attention_embed_dim,
                                                             params.combined_attention_num_heads,
                                                             params.combined_attention_dropout)
        self.combined_feedforward = FeedforwardNetwork(params.combined_feedforward)

        self.output_layers = nn.Sequential(
            OrderedDict([
                ('Linear', nn.Linear(params.combined_attention_embed_dim, params.combined_attention_embed_dim)),
                ('LeakyRelu', nn.LeakyReLU()),
                ('Norm', nn.LayerNorm(params.combined_attention_embed_dim))
            ])
        )

    def forward(self, x):
        audio_in, text_in = x

        audio_attention = self.audio_attention(audio_in, audio_in, audio_in)
        text_attention = self.text_attention(text_in, text_in, text_in)

        audio_text_attention = torch.cat([audio_attention, text_attention], dim=2)
        audio_text_attention = self.combined_attention_lin(audio_text_attention)

        out_mha, _ = self.combined_attention_main(audio_text_attention, audio_text_attention, audio_text_attention)
        out_mha = nn.functional.normalize(torch.add(out_mha, audio_text_attention))
        out_ff = self.combined_feedforward(out_mha)
        out = self.output_layers(out_ff)
        return out


class BCNetwork(nn.Module):
    def __init__(self, params, dyadic=False):
        super(BCNetwork, self).__init__()
        self.dyadic = dyadic
        self.bg_transformer = BGTransformer(params)
        self.fein_layer_main = FEIN(params.fein)
        if self.dyadic:
            self.fein_layer_inter = FEIN(params.fein)
        self.control_network = ControlNetwork(params.control_network, self.dyadic)

    def forward(self, x):
        if self.dyadic:
            return self.forward_dyadic(x)

        audio_in, text_in, max_len, absolut_pos, gesture = x

        audio_in += positional_encoding(audio_in.shape, max_len[0], absolut_pos=absolut_pos)
        text_in += positional_encoding(text_in.shape, max_len[0], absolut_pos=absolut_pos)

        gesture += positional_encoding(gesture.shape, max_len[0], absolut_pos=(absolut_pos - 100))
        input_x = audio_in, text_in

        gamma, beta = self.fein_layer_memory(torch.cat([audio_in, text_in], dim=1), gesture)
        output_combined = self.bg_transformer(input_x)

        return self.control_network(output_combined, gamma, beta)

    def forward_dyadic(self, x):
        audio_in, text_in, audio_in_inter, text_in_inter, max_len, absolut_pos, gesture, gesture_inter = x

        audio_in += positional_encoding(audio_in.shape, max_len[0], absolut_pos)
        text_in += positional_encoding(text_in.shape, max_len[0], absolut_pos)
        audio_in_inter += positional_encoding(audio_in_inter.shape, max_len[0], absolut_pos)
        text_in_inter += positional_encoding(text_in_inter.shape, max_len[0], absolut_pos)

        gesture += positional_encoding(gesture.shape, max_len[0], absolut_pos - 100)
        gesture_inter += positional_encoding(gesture_inter.shape, max_len[0], absolut_pos - 100)

        input_x = torch.cat([audio_in, audio_in_inter], dim=1), torch.cat([text_in, text_in_inter], dim=1)
        gamma, beta = self.fein_layer_main(torch.cat([audio_in, text_in], dim=1), gesture_inter)
        gamma_inter, beta_inter = self.fein_layer_inter(torch.cat([audio_in_inter, text_in_inter], dim=1), gesture)
        output_combined = self.bg_transformer(input_x)

        return self.control_network(output_combined, (gamma, gamma_inter), (beta, beta_inter))

    def inference(self, x, step_size=50, window_size=100):
        if self.dyadic:
            return self.inference_dyadic(x)
        audio_in, text_in, max_len, absolut_pos, pre_gesture_main = x
        padding = window_size - (audio_in.shape[2] % (window_size - step_size))

        audio_in += positional_encoding(audio_in.shape, max_len[0], absolut_pos)
        text_in += positional_encoding(text_in.shape, max_len[0], absolut_pos)
        audio_in = torch.cat([audio_in, torch.zeros((audio_in.shape[0], audio_in.shape[1], padding)).cuda()], dim=-1)
        text_in = torch.cat([text_in, torch.zeros((text_in.shape[0], text_in.shape[1], padding)).cuda()], dim=-1)
        pre_gesture_main += positional_encoding(pre_gesture_main.shape, max_len[0], absolut_pos - 50)

        iterations = int(audio_in.shape[2] / step_size)
        motion_out_main = []
        for i in range(iterations - 3):
            motion_out_main.append(pre_gesture_main)
            audio_in = audio_in[:, :, step_size * i: window_size + (step_size * i)]
            text_in = text_in[:, :, step_size * i: window_size + (step_size * i)]
            input_x = audio_in, text_in

            gamma, beta = self.fein_layer_main(torch.cat([audio_in, text_in], dim=1), pre_gesture_main)
            output_combined = self.bg_transformer(input_x)
            gestures_main = self.control_network(output_combined, gamma, beta)
            pre_gesture_main = gestures_main.transpose(1, 2)[:, :, window_size - 50:]
        motion_out_main.append(pre_gesture_main.transpose(1, 2)[:, :, :window_size - padding])
        return torch.cat(motion_out_main, dim=2)

    def inference_dyadic(self, x, step_size=50, window_size=50):
        audio_in, text_in, audio_in_inter, text_in_inter, max_len, absolut_pos, pre_gesture_main, pre_gesture_inter = x
        # padding = window_size - (audio_in.shape[2] % window_size)

        # audio_in += positional_encoding(audio_in.shape, max_len[0], absolut_pos)
        # text_in += positional_encoding(text_in.shape, max_len[0], absolut_pos)
        # audio_in_inter += positional_encoding(audio_in_inter.shape, max_len[0], absolut_pos)
        # text_in_inter += positional_encoding(text_in_inter.shape, max_len[0], absolut_pos)

        # audio_in = torch.cat([audio_in, torch.zeros((audio_in.shape[0], audio_in.shape[1], padding)).cuda()], dim=-1)
        # text_in = torch.cat([text_in, torch.zeros((text_in.shape[0], text_in.shape[1], padding)).cuda()], dim=-1)
        # audio_in_inter = torch.cat(
        #     [audio_in_inter, torch.zeros((audio_in.shape[0], audio_in.shape[1], padding)).cuda()], dim=-1)
        # text_in_inter = torch.cat([text_in_inter, torch.zeros((text_in.shape[0], text_in.shape[1], padding)).cuda()],
        #                           dim=-1)

        # iterations = int(audio_in.shape[2] / step_size)
        # motion_out_main = []
        # motion_out_inter = []
        # for i in range(iterations - 1):
        #     motion_out_main.append(pre_gesture_main)
        #     motion_out_inter.append(pre_gesture_inter)

        #     pre_gesture_main += positional_encoding(pre_gesture_main.shape, step_size, absolut_pos - 50)
        #     pre_gesture_inter += positional_encoding(pre_gesture_inter.shape, step_size, absolut_pos - 50)

        #     audio_input = audio_in[:, :, step_size * i: window_size + (step_size * i)]
        #     text_input = text_in[:, :, step_size * i: window_size + (step_size * i)]
        #     audio_input_inter = audio_in_inter[:, :, step_size * i: window_size + (step_size * i)]
        #     text_input_inter = text_in_inter[:, :, step_size * i: window_size + (step_size * i)]

        #     absolut_pos += step_size

        #     input_x = torch.cat([audio_input, audio_input_inter], dim=1), torch.cat([text_input, text_input_inter],
        #                                                                             dim=1)
        #     gamma, beta = self.fein_layer_main(torch.cat([audio_input, text_input], dim=1), pre_gesture_inter)
        #     gamma_inter, beta_inter = self.fein_layer_inter(torch.cat([audio_input_inter, text_input_inter], dim=1),
        #                                                     pre_gesture_main)
        #     output_combined = self.bg_transformer(input_x)

        #     gestures_main, gesture_inter = self.control_network(output_combined, (gamma, gamma_inter),
        #                                                         (beta, beta_inter))
        #     pre_gesture_main = gestures_main.transpose(1, 2)
        #     pre_gesture_inter = gesture_inter.transpose(1, 2)
        # motion_out_main.append(pre_gesture_main[:, :, :window_size - padding])
        # motion_out_inter.append(pre_gesture_inter[:, :, :window_size - padding])

        return torch.rand((pre_gesture_main.shape[0], pre_gesture_main.shape[1], 336)) #torch.cat([torch.cat(motion_out_main, dim=2), torch.cat(motion_out_inter, dim=2)], dim=1)

    def validate_dyadic(self, x):
        criterion = BCGLoss({'joints': 10, 'velocity': 5, 'acceleration': 2})
        audio_in, text_in, audio_in_inter, text_in_inter, max_len, gesture_main, gesture_inter = x
        audio_in = audio_in[:, :, 50:]
        text_in = text_in[:, :, 50:]
        audio_in_inter = audio_in_inter[:, :, 50:]
        text_in_inter = text_in_inter[:, :, 50:]
        pre_gesture_main = gesture_main[:, :, :50]
        pre_gesture_inter = gesture_inter[:, :, :50]
        padding = 50 - (audio_in.shape[2] % 50)
        absolut_pos = torch.FloatTensor([50])
        max_len += padding - 50
        audio_in = torch.cat([audio_in, torch.zeros((audio_in.shape[0], audio_in.shape[1], padding)).cuda()], dim=-1)
        text_in = torch.cat([text_in, torch.zeros((text_in.shape[0], text_in.shape[1], padding)).cuda()], dim=-1)
        audio_in_inter = torch.cat(
            [audio_in_inter, torch.zeros((audio_in.shape[0], audio_in.shape[1], padding)).cuda()], dim=-1)
        text_in_inter = torch.cat([text_in_inter, torch.zeros((text_in.shape[0], text_in.shape[1], padding)).cuda()],
                                  dim=-1)

        audio_in += positional_encoding(audio_in.shape, max_len, absolut_pos)
        text_in += positional_encoding(text_in.shape, max_len, absolut_pos)
        audio_in_inter += positional_encoding(audio_in_inter.shape, max_len, absolut_pos)
        text_in_inter += positional_encoding(text_in_inter.shape, max_len, absolut_pos)

        motion_out_main = []
        motion_out_inter = []
        idx = 0
        for i in range(50, max_len + padding, 50):
            motion_out_main.append(pre_gesture_main)
            motion_out_inter.append(pre_gesture_inter)
            # pre_gesture_main = gesture_main[:, :, (i - 50): i]
            # pre_gesture_inter = gesture_inter[:, :, (i - 50): i]
            pre_gesture_main += positional_encoding(pre_gesture_main.shape, 50, absolut_pos)
            pre_gesture_inter += positional_encoding(pre_gesture_inter.shape, 50, absolut_pos)
            absolut_pos += 50
            audio_input = audio_in[:, :, idx:i]
            audio_input_inter = audio_in_inter[:, :, idx:i]
            text_input = text_in[:, :, idx:i]
            text_input_inter = text_in_inter[:, :, idx:i]
            idx += 50
            input_x = torch.cat([audio_input, audio_input_inter], dim=1), torch.cat([text_input, text_input_inter],
                                                                                    dim=1)
            gamma, beta = self.fein_layer_main(torch.cat([audio_input, text_input], dim=1), pre_gesture_inter)
            gamma_inter, beta_inter = self.fein_layer_inter(torch.cat([audio_input_inter, text_input_inter], dim=1),
                                                            pre_gesture_main)
            output_combined = self.bg_transformer(input_x)

            pred_gesture_main, pred_gesture_inter = self.control_network(output_combined, (gamma, gamma_inter),
                                                                         (beta, beta_inter))
            # print(
            #     f'loss in :{criterion(torch.cat([pred_gesture_main, pred_gesture_inter]), torch.cat([gesture_main[:, :, i:i + 50], gesture_inter[:, :, i:i + 50]]))}')
            pre_gesture_main = pred_gesture_main.transpose(1, 2)
            pre_gesture_inter = pred_gesture_inter.transpose(1, 2)
        motion_out_main.append(pre_gesture_main[:, :, :50 - padding])
        motion_out_inter.append(pre_gesture_inter[:, :, :50 - padding])
        return torch.cat(motion_out_main, dim=2), torch.cat(motion_out_inter, dim=2)


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
