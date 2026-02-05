import torch
import torch.nn as nn
import torch.nn.functional as F
from .FiLM import FiLM

"""
B : batch dim
F or C : feature, frequency, channel dim
E : embedding dim
T : time dim
"""

class ConvNormPReLU(nn.Module):
    """
    Conv1d -> LayerNorm -> PReLU
    """
    def __init__(self, in_dim, out_dim, kernel_size=1, chomp_size=0, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, **kwargs)
        self.ln = nn.LayerNorm(out_dim)
        self.prelu = nn.PReLU()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x : (B, in_dim, T)
        Returns:
            x : (B, out_dim, T)
        """
        x = self.conv(x)
        if self.chomp_size>0: x = x[:,:,:-self.chomp_size]
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = self.prelu(x)
        return x


class TCNNBlock(nn.Module):
    def __init__(self, in_dim, hid_dim=None, kernel_size=3, dilation=1, causal=False):
        super().__init__()

        if hid_dim==None: hid_dim = in_dim // 4
        self.causal = causal
        if self.causal:
            self.padding_size = (kernel_size-1)*dilation
            self.chomp_size = self.padding_size
        else:
            self.padding_size = (kernel_size-1)*dilation//2  # ='same'
            self.chomp_size = 0

        self.conv_in = ConvNormPReLU(in_dim, hid_dim, 1)
        self.conv_d = ConvNormPReLU(hid_dim, hid_dim, kernel_size,
                                    chomp_size=self.chomp_size,
                                    padding=self.padding_size,
                                    dilation=dilation, bias=False)
        self.conv_out = ConvNormPReLU(hid_dim, in_dim, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x : (B, in_dim, T)
        Returns:
            x : (B, in_dim, T)
        """
        identity = x
        out = self.conv_in(x)
        out = self.conv_d(out)
        out = self.conv_out(out)
        out = identity + out
        return out


class TCNN(nn.Module):
    def __init__(self, in_dim=25, out_dim=64, init_dilation=2, num_layers=5):
        super().__init__()
        self.conv_in = ConvNormPReLU(in_dim, out_dim, 1)
        self.blocks = nn.ModuleList([TCNNBlock(out_dim, dilation=init_dilation**i, causal=True) for i in range(num_layers)])

    def forward(self, x):
        """
        Args:
            x : (B, in_dim, T)
        Returns:
            x : (B, out_dim, T)
        """
        x = self.conv_in(x)
        for block in self.blocks: x = block(x)
        return x




class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim, out_dim, att_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, att_dim, kernel_size=1),
            nn.BatchNorm1d(att_dim),
            nn.ReLU(), #nn.Tanh(),
            nn.Conv1d(att_dim, 1, kernel_size=1),
        )
        self.proj = nn.Linear(in_dim*2, out_dim)

    def forward(self, x, lengths=None):
        """
        Args:
            x : (B, in_dim, T)
        Returns:
            x : (B, out_dim)
        """
        alpha = self.attention(x)  # (B, 1, T)
        if lengths is not None:
            mask = torch.arange(x.size(2), device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(1)
            alpha.masked_fill_(~mask, float('-inf'))
        alpha = F.softmax(alpha, dim=2)

        mean = torch.sum(alpha * x, dim=2)  # (B, D)
        mean_sq = torch.sum(alpha * (x**2), dim=2)  # (B, D)
        std = torch.sqrt((mean_sq - mean ** 2).clamp(min=1e-7))  # (B, D)
        stats = torch.cat((mean, std), dim=1)  # (B, 2*D)
        out = self.proj(stats)
        return out

    
class EmbeddingExtractor(nn.Module):
    def __init__(self, in_dim=25, out_dim=80, dropout_p=0.3, num_layers=5, att_dim=128):
        super().__init__()
        self.tcnn = TCNN(in_dim, out_dim, num_layers=num_layers)
        self.pooling = AttentiveStatsPooling(out_dim, out_dim, att_dim=att_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, lengths=None):
        """
        Args:
            x : (B, in_dim, T)
            lengths : (B, )
        Returns:
            x : (B, out_dim)
        """
        out = self.tcnn(x)
        out = self.pooling(out, lengths)
        out = self.dropout(out)
        return out


class AAM(nn.Module):
    def __init__(self, in_dim, out_dim, margin=0.2, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        """
        Args:
            x : (B, in_dim)
            label : (B, )
        Returns:
            x : (B, out_dim)
        """
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if label is not None:
            phi = cosine - self.margin
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            out = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            out = cosine
        out *= self.scale
        return out




class PClassifier(nn.Module):
    def __init__(self, in_dim, dropout_p=0.2, hid_dim1=40, hid_dim2=20):
        super().__init__()
        self.conv1 = ConvNormPReLU(in_dim, hid_dim1, 7, padding='same')
        self.conv2 = ConvNormPReLU(hid_dim1, hid_dim2, 5, padding='same')
        self.dropout = nn.Dropout(p=dropout_p)
        self.final_linear = nn.Conv1d(hid_dim2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_dim, T)
        Returns:
            logit : (B, T)
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.final_linear(out)
        out = out.squeeze(1)
        return out


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, feat, emb):
        emb_repeated = emb.unsqueeze(2).expand(-1, -1, feat.size(2))
        out = torch.cat((feat, emb_repeated), dim=1)
        return out

class FiLM_wrapper(FiLM):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(input_dim, embedding_dim)
    def forward(self, x, embedding):
        out = super().forward(x.transpose(1,2), embedding)
        return out.transpose(1,2)

class AttentiveScore(nn.Module):
    def __init__(self, feat_dim, emb_dim, modulation='Concat', keep_orig=False, chunk_size=None, return_score=True):
        super().__init__()
        assert modulation in ['Concat','FiLM']
        self.modulation = modulation
        if modulation=='Concat':
            self.modulation = Concat()
            self.first_dim = feat_dim+emb_dim
        elif modulation=='FiLM':
            self.modulation = FiLM_wrapper(input_dim=feat_dim, embedding_dim=emb_dim)
            self.first_dim = feat_dim
        else: raise Exception('not supported modulation')
        if keep_orig==True: raise Exception("does not support keep_orig=True anymore.")
        self.chunk_size = chunk_size
        self.return_score = return_score
        self.conv1 = ConvNormPReLU(self.first_dim, feat_dim, 1)
        self.conv2 = nn.Conv1d(feat_dim, feat_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        if return_score:
            self.loss_linear = nn.Conv1d(feat_dim, 1, 1)
            self.loss_sigmoid = nn.Sigmoid()

    def forward(self, feat: torch.Tensor, emb: torch.Tensor, lengths=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat : (B, feat_dim, T)
            emb : (B, emb_dim)
        Returns:
            feat_as : (B, feat_dim, T)
            score_normalized : (B, T)
        """
        B, C, T = feat.shape
        modulated = self.modulation(feat, emb)
        modulated_proj = self.conv1(modulated)
        score = self.conv2(modulated_proj)
        if lengths is not None:
            mask = torch.arange(feat.size(2), device=score.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B,T)
            mask = mask.unsqueeze(1)  # (B,1,T)
            score = score.masked_fill(~mask, float("-inf"))
        if self.chunk_size is None:
            score = self.softmax(score)
        else:
            chunk_size = self.chunk_size
            num_chunks = (T + chunk_size - 1) // chunk_size
            pad_len = num_chunks * chunk_size - T
            score_padded = F.pad(score, (0, pad_len), value=float('-inf'))
            score_chunks = score_padded.view(B, C, num_chunks, chunk_size)
            score_chunks = self.softmax(score_chunks)
            score = score_chunks.view(B, C, -1)[..., :T]
        score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
        #feat_as = feat*score if self.keep_orig else modulated*score
        feat_as = modulated*score if self.modulation=='Concat' else modulated_proj*score

        if self.return_score:
            score_normalized = self.loss_linear(score)
            score_normalized = self.loss_sigmoid(score_normalized)
            score_normalized = score_normalized.squeeze(1)
        else: score_normalized = torch.zeros((B, T), device=score.device, dtype=score.dtype)

        return feat_as, score_normalized


class ASPVAD(nn.Module):
    def __init__(self, in_dim=24, num_speakers=128, modulation='Concat',
                 emb_dim=80, emb_num_layers=5, feat_num_layers=5, att_dim=128, p_hid_dim1=40, p_hid_dim2=20,
                 with_train_layers=True, dropout_p=0, AS_keep_orig=False, AS_chunk_size=None, pred_spk=True):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.with_train_layers = with_train_layers
        self.num_speakers = num_speakers
        self.pred_spk = pred_spk & with_train_layers
        self.embedding_extractor = EmbeddingExtractor(in_dim, emb_dim, dropout_p=dropout_p, num_layers=emb_num_layers, att_dim=att_dim)
        self.feature_tcnn = TCNN(in_dim, emb_dim, num_layers=feat_num_layers)
        self.attentive_score = AttentiveScore(emb_dim, emb_dim, modulation=modulation, keep_orig=AS_keep_orig, chunk_size=AS_chunk_size, return_score=with_train_layers)
        self.p_classifier = PClassifier(emb_dim, dropout_p=dropout_p, hid_dim1=p_hid_dim1, hid_dim2=p_hid_dim2)
        if self.pred_spk: self.aam = AAM(emb_dim, num_speakers)
        

    def forward(self, enroll_feat, simul_feat, enroll_length=None, simul_length=None, spk_label=None):
        """
        Args:
            enroll_feat : (B, F, T')
            simul_feat : (B, F, T)
        Returns:
            pvad_logit : (B, T)
            score : (B, T)
            spk_logit : (B, num_speakers)
        """
        
        enroll_emb = self.embedding_extractor(enroll_feat, enroll_length) # (B, E)
        simul_feat = self.feature_tcnn(simul_feat) # (B, E, T)
        feat_as, score = self.attentive_score(simul_feat, enroll_emb, simul_length)
        pvad_logit = self.p_classifier(feat_as)
        spk_logit = self.aam(enroll_emb, spk_label) if self.pred_spk else torch.zeros((pvad_logit.size(0), self.num_speakers), device=pvad_logit.device, dtype=pvad_logit.dtype)
        
        return pvad_logit, score, spk_logit



class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = F.cross_entropy

    def forward(self, p1, s, spk_logit, q1, speaker_labels):
        valid_mask = (q1 != -1)
        p1_masked = p1[valid_mask]
        s_masked = s[valid_mask]
        q1_masked = q1[valid_mask]

        L_pVAD = self.bce_loss(p1_masked, q1_masked)
        L_AS = self.mse_loss(s_masked, q1_masked)
        L_spkid = self.cross_entropy(spk_logit, speaker_labels)

        L_total = L_pVAD + L_AS + L_spkid
        return L_total, L_pVAD, L_AS, L_spkid




