import torch 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from utils import cc
from utils import pad_list
from utils import _seq_mask
from utils import _inflate
from utils import _inflate_np
from utils import weight_init
from torch.distributions.categorical import Categorical
import os
import copy

def _get_vgg2l_odim(idim, in_channel=1, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)
    return int(idim) * out_channel

def _pad_one_frame(inp):
    inp_t = inp.transpose(1, 2)
    out_t = F.pad(inp_t, (0, 1), mode='replicate')
    out = out_t.transpose(1, 2)
    return out

class VGG2L(torch.nn.Module):
    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        self.in_channel = in_channel
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

    def conv_block(self, inp, layers):
        out = inp
        for layer in layers:
            out = F.relu(layer(out))
        out = F.max_pool2d(out, 2, stride=2, ceil_mode=True)
        return out

    def forward(self, xs, ilens):
        # xs = [batch_size, frames, feeature_dim]
        # ilens is a list of frame length of each utterance 
        xs = torch.transpose(
                xs.view(xs.size(0), xs.size(1), self.in_channel, xs.size(2)//self.in_channel), 1, 2)
        xs = self.conv_block(xs, [self.conv1_1, self.conv1_2])
        xs = self.conv_block(xs, [self.conv2_1, self.conv2_2])
        ilens = np.array(np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64) 
        ilens = np.array(np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()
        xs = torch.transpose(xs, 1, 2)
        xs = xs.contiguous().view(xs.size(0), xs.size(1), xs.size(2) * xs.size(3))
        return xs, ilens

class pBLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate):
        super(pBLSTM, self).__init__()
        layers, project_layers = [], []
        for i in range(n_layers):
            #idim = input_dim if i == 0 else hidden_dim
            idim = input_dim if i == 0 else hidden_dim
            project_dim = hidden_dim * 4 if subsample[i] > 1 else hidden_dim * 2

            layers.append(torch.nn.LSTM(idim, hidden_dim, num_layers=1,
                bidirectional=True, batch_first=True))

            project_layers.append(torch.nn.Linear(project_dim, hidden_dim))
        self.layers = torch.nn.ModuleList(layers)
        self.project_layers = torch.nn.ModuleList(project_layers)
        self.dropout_layer = torch.nn.Dropout(p=dropout_rate)
        self.subsample = subsample

    def forward(self, xpad, ilens):
        for i, (layer, project_layer) in enumerate(zip(self.layers, self.project_layers)):
            # pack sequence 
            xs_pack = pack_padded_sequence(xpad, ilens, batch_first=True)
            ys, (_, _) = layer(xs_pack)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            ys_pad = self.dropout_layer(ys_pad)
            ilens = ilens.numpy()
            # subsampling
            sub = self.subsample[i]
            if sub > 1:
                # pad one frame
                if ys_pad.size(1) % 2 == 1:
                    ys_pad = F.pad(ys_pad.transpose(1, 2), (0, 1), mode='replicate').transpose(1, 2)
                # concat two frames
                ys_pad = ys_pad.contiguous().view(ys_pad.size(0), ys_pad.size(1) // 2, ys_pad.size(2) * 2)
                ilens = [(length + 1) // sub for length in ilens]
            projected = project_layer(ys_pad)
            xpad = F.relu(projected)
            xpad = self.dropout_layer(xpad)
        # type to list of int
        ilens = np.array(ilens, dtype=np.int64).tolist()
        return xpad, ilens

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate, in_channel=1, vgg=False):
        super(Encoder, self).__init__()
        self.vgg = vgg
        if vgg:
            self.enc1 = VGG2L(in_channel)
            out_channels = _get_vgg2l_odim(input_dim)
            self.enc2 = pBLSTM(input_dim=out_channels, hidden_dim=hidden_dim, 
                    n_layers=n_layers, subsample=subsample, dropout_rate=dropout_rate)
        else:
            self.enc = pBLSTM(input_dim=input_dim, hidden_dim=hidden_dim, 
                    n_layers=n_layers, subsample=subsample, dropout_rate=dropout_rate)

    def forward(self, x, ilens):
        if self.vgg:
            out, ilens = self.enc1(x, ilens)
            out, ilens = self.enc2(out, ilens)
        else:
            out, ilens = self.enc(x, ilens)
        return out, ilens

class AttLoc(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim, conv_channels, conv_kernel_size, att_odim):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(encoder_dim, att_dim)
        self.mlp_dec = torch.nn.Linear(decoder_dim, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(conv_channels, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(1, conv_channels, (1, 2 * conv_kernel_size + 1), 
                padding=(0, conv_kernel_size), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1, bias=False)
        self.mlp_o = torch.nn.Linear(encoder_dim, att_odim)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.att_dim = att_dim
        self.att_odim = att_odim
        self.conv_channels = conv_channels
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_pad, enc_len, dec_z, att_prev, scaling=2.0):
        batch_size =enc_pad.size(0)
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_pad
            self.enc_length = self.enc_h.size(1)
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_pad.new_zeros(batch_size, self.decoder_dim)
        else:
            dec_z = dec_z.view(batch_size, self.decoder_dim)

        if att_prev is None:
            # initialize attention weights to uniform
            att_prev = pad_list([self.enc_h.new(l).fill_(1.0 / l) for l in enc_len], 0)

        #att_prev: batch_size x frame
        att_conv = self.loc_conv(att_prev.view(batch_size, 1, 1, self.enc_length))
        # att_conv: batch_size x channel x 1 x frame -> batch_size x frame x channel
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: batch_size x frame x channel -> batch_size x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: batch_size x 1 x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch_size, 1, self.att_dim)
        att_state = torch.tanh(self.pre_compute_enc_h + dec_z_tiled + att_conv)
        e = self.gvec(att_state).squeeze(2)
        # w: batch_size x frame
        w = F.softmax(scaling * e, dim=1)
        # w_expanded: batch_size x 1 x frame
        w_expanded = w.unsqueeze(1)
        #c = torch.sum(self.enc_h * w_expanded, dim=1)
        c = torch.bmm(w_expanded, self.enc_h).squeeze(1)
        c = self.mlp_o(c)
        return c, w

class MultiHeadAttLoc(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim, conv_channels, conv_kernel_size, heads, att_odim):
        super(MultiHeadAttLoc, self).__init__()
        self.heads = heads
        self.mlp_enc = torch.nn.ModuleList([torch.nn.Linear(encoder_dim, att_dim) for _ in range(self.heads)])
        self.mlp_dec = torch.nn.ModuleList([torch.nn.Linear(decoder_dim, att_dim, bias=False) \
                for _ in range(self.heads)])
        self.mlp_att = torch.nn.ModuleList([torch.nn.Linear(conv_channels, att_dim, bias=False) \
                for _ in range(self.heads)])
        self.loc_conv = torch.nn.ModuleList([torch.nn.Conv2d(
                1, conv_channels, (1, 2 * conv_kernel_size + 1), 
                padding=(0, conv_kernel_size), bias=False) for _ in range(self.heads)])
        self.gvec = torch.nn.ModuleList([torch.nn.Linear(att_dim, 1, bias=False) for _ in range(self.heads)])
        self.mlp_o = torch.nn.Linear(self.heads * encoder_dim, att_odim)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.att_dim = att_dim
        self.conv_channels = conv_channels
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        self.enc_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_pad, enc_len, dec_z, att_prev, scaling=2.0):
        batch_size =enc_pad.size(0)
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_pad
            self.enc_length = self.enc_h.size(1)
            self.pre_compute_enc_h = [self.mlp_enc[h](self.enc_h) for h in range(self.heads)]

        if dec_z is None:
            dec_z = enc_pad.new_zeros(batch_size, self.decoder_dim)
        else:
            dec_z = dec_z.view(batch_size, self.decoder_dim)

        # initialize attention weights to uniform
        if att_prev is None:
            att_prev = []
            for h in range(self.heads):
                att_prev += [pad_list([self.enc_h.new(l).fill_(1.0 / l) for l in enc_len], 0)]

        cs, ws = [], []
        for h in range(self.heads):
            #att_prev: batch_size x frame
            att_conv = self.loc_conv[h](att_prev[h].view(batch_size, 1, 1, self.enc_length))
            # att_conv: batch_size x channel x 1 x frame -> batch_size x frame x channel
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            # att_conv: batch_size x frame x channel -> batch_size x frame x att_dim
            att_conv = self.mlp_att[h](att_conv)

            # dec_z_tiled: batch_size x 1 x att_dim
            dec_z_tiled = self.mlp_dec[h](dec_z).view(batch_size, 1, self.att_dim)
            att_state = torch.tanh(self.pre_compute_enc_h[h] + dec_z_tiled + att_conv)
            e = self.gvec[h](att_state).squeeze(2)
            # w: batch_size x frame
            w = F.softmax(scaling * e, dim=1)
            ws.append(w)
            # w_expanded: batch_size x 1 x frame
            w_expanded = w.unsqueeze(1)
            #c = torch.sum(self.enc_h * w_expanded, dim=1)
            c = torch.bmm(w_expanded, self.enc_h).squeeze(1)
            cs.append(c)
        c = self.mlp_o(torch.cat(cs, dim=1))
        return c, ws 

class Decoder(torch.nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, attention, att_odim, 
            dropout_rate, bos, eos, pad, ls_weight=0, labeldist=None):
        super(Decoder, self).__init__()
        self.bos, self.eos, self.pad = bos, eos, pad
        self.embedding = torch.nn.Embedding(output_dim, embedding_dim, padding_idx=pad)
        self.LSTMCell = torch.nn.LSTMCell(embedding_dim + att_odim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim + att_odim, output_dim)
        self.dropout_layer = torch.nn.Dropout(p=dropout_rate)
        self.attention = attention

        self.hidden_dim = hidden_dim
        self.att_odim = att_odim
        self.dropout_rate = dropout_rate

        # label smoothing hyperparameters
        self.ls_weight = ls_weight
        self.labeldist = labeldist
        if labeldist is not None:
            self.vlabeldist = cc(torch.from_numpy(np.array(labeldist, dtype=np.float32)))

    def zero_state(self, enc_pad, dim=None):
        if not dim:
            return enc_pad.new_zeros(enc_pad.size(0), self.hidden_dim)
        else:
            return enc_pad.new_zeros(enc_pad.size(0), dim)

    def forward_step(self, emb, dec_z, dec_c, c, w, enc_pad, enc_len):
        cell_inp = torch.cat([emb, c], dim=-1)
        cell_inp = self.dropout_layer(cell_inp)
        dec_z, dec_c = self.LSTMCell(cell_inp, (dec_z, dec_c))

        # run attention module
        c, w = self.attention(enc_pad, enc_len, dec_z, w)
        output = torch.cat([dec_z, c], dim=-1)
        #output = self.dropout_layer(output)
        #output = F.dropout(output, self.dropout_rate)
        logit = self.output_layer(output)
        return logit, dec_z, dec_c, c, w

    def forward(self, enc_pad, enc_len, ys=None, tf_rate=1.0, max_dec_timesteps=500, 
            sample=False, label_smoothing=True):
        batch_size = enc_pad.size(0)
        if ys is not None:
            # prepare input and output sequences
            bos = ys[0].data.new([self.bos])
            eos = ys[0].data.new([self.eos])
            ys_in = [torch.cat([bos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, eos], dim=0) for y in ys]
            pad_ys_in = pad_list(ys_in, pad_value=self.eos)
            pad_ys_out = pad_list(ys_out, pad_value=self.eos)
            # get length info
            batch_size, olength = pad_ys_out.size(0), pad_ys_out.size(1)
            # map idx to embedding
            eys = self.embedding(pad_ys_in)

        # initialization
        dec_c = self.zero_state(enc_pad)
        dec_z = self.zero_state(enc_pad)
        c = self.zero_state(enc_pad, dim=self.att_odim)

        w = None
        logits, prediction, ws = [], [], []
        # reset the attention module
        self.attention.reset()

        # loop for each timestep
        olength = max_dec_timesteps if not ys else olength
        for t in range(olength):
            # supervised learning: using teacher forcing
            if ys is not None:
                # teacher forcing
                tf = True if np.random.random_sample() <= tf_rate else False
                emb = eys[:, t, :] if tf or t == 0 else self.embedding(prediction[-1])
            # else, label the data with greedy/sampling
            else:
                if t == 0:
                    bos = cc(torch.Tensor([self.bos for _ in range(batch_size)]).type(torch.LongTensor))
                    emb = self.embedding(bos)
                else:
                    emb = self.embedding(prediction[-1])
            logit, dec_z, dec_c, c, w = \
                    self.forward_step(emb, dec_z, dec_c, c, w, enc_pad, enc_len)

            ws.append(w)
            logits.append(logit)
            if not sample:
                prediction.append(torch.argmax(logit, dim=-1))
            else:
                sampled_indices = Categorical(logits=logit).sample() 
                prediction.append(sampled_indices)

        logits = torch.stack(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=2)
        prediction = torch.stack(prediction, dim=1)
        ws = torch.stack(ws, dim=1)
        if ys:
            ys_log_probs = torch.gather(log_probs, dim=2, index=pad_ys_out.unsqueeze(2)).squeeze(2)
        else:
            ys_log_probs = torch.gather(log_probs, dim=2, index=prediction.unsqueeze(2)).squeeze(2)

        # label smoothing
        if label_smoothing and self.ls_weight > 0 and self.training:
            loss_reg = torch.sum(log_probs * self.vlabeldist, dim=2)
            ys_log_probs = (1 - self.ls_weight) * ys_log_probs + self.ls_weight * loss_reg
        return logits, ys_log_probs, prediction, ws

    def recognize_beams(self, enc_pad, enc_len, max_dec_timesteps, topk):
        pass

class E2E(torch.nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, enc_n_layers, subsample, dropout_rate, 
            dec_hidden_dim, att_dim, conv_channels, conv_kernel_size, att_odim,
            embedding_dim, output_dim, ls_weight, labeldist, vgg=False, 
            pad=0, bos=1, eos=2):

        super(E2E, self).__init__()

        # encoder to encode acoustic features
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=enc_hidden_dim, 
                n_layers=enc_n_layers, subsample=subsample, dropout_rate=dropout_rate, vgg=vgg)

        # attention module
        self.attention = AttLoc(encoder_dim=enc_hidden_dim, 
                decoder_dim=dec_hidden_dim, att_dim=att_dim, 
                conv_channels=conv_channels, conv_kernel_size=conv_kernel_size, 
                att_odim=att_odim)

        # decoder to generate words (or other units)
        self.decoder = Decoder(output_dim=output_dim, 
                hidden_dim=dec_hidden_dim, 
                embedding_dim=embedding_dim,
                attention=self.attention, 
                dropout_rate=dropout_rate, 
                att_odim=att_odim, 
                ls_weight=ls_weight, 
                labeldist=labeldist, 
                bos=bos, 
                eos=eos, 
                pad=pad)

    def forward(self, data, ilens, ys=None, tf_rate=1.0, max_dec_timesteps=200, 
            sample=False, label_smoothing=True):
        enc_h, enc_lens = self.encoder(data, ilens)
        logits, log_probs, prediction, ws = self.decoder(enc_h, enc_lens, ys, 
                tf_rate=tf_rate, max_dec_timesteps=max_dec_timesteps, 
                sample=sample, label_smoothing=label_smoothing)
        return logits, log_probs, prediction, ws

    def mask_and_cal_loss(self, log_probs, ys, mask=None):
        # add 1 to EOS
        if mask is None: 
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
        else:
            seq_len = [y.size(0) for y in ys]
        # divide by total length
        loss = -torch.sum(log_probs * mask) / sum(seq_len)
        return loss

class MultiDecE2E(torch.nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, enc_n_layers, subsample, dropout_rate, 
            dec_hidden_dim, att_dim, conv_channels, conv_kernel_size, att_odim,
            embedding_dim, output_dim, ls_weight, labeldist, vgg=False, 
            pad=0, bos=1, eos=2):

        super(MultiDecE2E, self).__init__()

        # encoder to encode acoustic features
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=enc_hidden_dim, 
                n_layers=enc_n_layers, subsample=subsample, dropout_rate=dropout_rate, vgg=vgg)

        # attention module
        self.attention = AttLoc(encoder_dim=enc_hidden_dim, 
                decoder_dim=dec_hidden_dim, att_dim=att_dim, 
                conv_channels=conv_channels, conv_kernel_size=conv_kernel_size, 
                att_odim=att_odim)

        # decoder to generate words (or other units)
        self.decoder = Decoder(output_dim=output_dim, 
                hidden_dim=dec_hidden_dim, 
                embedding_dim=embedding_dim,
                attention=self.attention, 
                dropout_rate=dropout_rate, 
                att_odim=att_odim, 
                ls_weight=ls_weight, 
                labeldist=labeldist, 
                bos=bos, 
                eos=eos, 
                pad=pad)

        # the same architecture for the second decoder, share the attention parameters
        self.aux_decoder = Decoder(output_dim=output_dim, 
                hidden_dim=dec_hidden_dim, 
                embedding_dim=embedding_dim,
                attention=self.attention, 
                dropout_rate=dropout_rate, 
                att_odim=att_odim, 
                ls_weight=ls_weight, 
                labeldist=labeldist, 
                bos=bos, 
                eos=eos, 
                pad=pad)

    def forward(self, data, ilens, ys=None, tf_rate=1.0, max_dec_timesteps=200, 
            sample=False, label_smoothing=True, enc_only=False, dec_only=False):
        with torch.no_grad() if dec_only else torch.enable_grad():
            enc_h, enc_lens = self.encoder(data, ilens)
        with torch.no_grad() if enc_only else torch.enable_grad():
            # probabilities from the first decoder
            logits, log_probs, prediction, ws = self.decoder(enc_h, enc_lens, ys, 
                    tf_rate=tf_rate, max_dec_timesteps=max_dec_timesteps, 
                    sample=sample, label_smoothing=label_smoothing)
            # probabilities from the second decoder
            if ys is not None:
                aux_logits, aux_log_probs, aux_prediction, aux_ws = self.aux_decoder(enc_h, enc_lens, ys=ys, 
                        tf_rate=tf_rate, max_dec_timesteps=max_dec_timesteps, 
                        sample=sample, label_smoothing=label_smoothing)
            else:
                # using prediction from the first decoder as the input 
                aux_logits, aux_log_probs, aux_prediction, aux_ws = self.aux_decoder(enc_h, enc_lens, 
                        ys=prediction, 
                        tf_rate=tf_rate, max_dec_timesteps=max_dec_timesteps, 
                        sample=sample, label_smoothing=label_smoothing)
        return (logits, log_probs, prediction, ws), (aux_logits, aux_log_probs, aux_predictions, aux_ws)

    def mask_and_cal_loss(self, log_probs, ys, mask=None):
        # add 1 to EOS
        if mask is None: 
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
        else:
            seq_len = [y.size(0) for y in ys]
        # divide by total length
        loss = -torch.sum(log_probs * mask) / sum(seq_len)
        return loss
