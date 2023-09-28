import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class Traj_embedding(nn.Module):
    """
    生成每个行人的轨迹点token
    """
    def __init__(self,embedding_dim=64):
        super(Traj_embedding, self).__init__()
        self.embbedding_dim = embedding_dim
        self.tra_embedding = nn.Linear(2, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
    def forward(self, obs_traj):
        batch = obs_traj.size(1)
        obs_traj_embedding = self.tra_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embbedding_dim)
        obs_traj_embedding = self.norm(obs_traj_embedding)
        return obs_traj_embedding
    
class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, drop_rate=0.):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act_layer = nn.LeakyReLU()#nn.ReLU()
        self.drop = nn.Dropout(drop_rate)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Transformer_Encoder(nn.Module):
    """
    对每个行人序列加自注意力，这里的batchsize = batchsize*每组数据的行人数量
    """
    def __init__(self,input_dim, output_dim, mlp_hid_dim, num_head=1, obs_len=16, drop_rate=0.):
        super(Transformer_Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.obs_len = obs_len
        self.pos_emb = nn.Parameter(torch.zeros(1, self.obs_len, self.input_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.q = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.kv = nn.Linear(self.input_dim, self.output_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(p=drop_rate)
        self.proj = nn.Linear(output_dim, output_dim)
        self.proj_drop = nn.Dropout(p=drop_rate)
        trunc_normal_(self.pos_emb, std=.02)
        self.mlp = Mlp(output_dim, mlp_hid_dim, output_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, obs_traj_embedding):
        obs_traj_embedding = obs_traj_embedding + self.pos_emb
        obs_traj_embedding = self.pos_drop(obs_traj_embedding) #675*16*64
        obs_traj_embedding = self.norm1(obs_traj_embedding)
        N, L, dim = obs_traj_embedding.shape#N一个batchsize中的序列数，L输入的序列长度，dim嵌入的特征维度
        q = self.q(obs_traj_embedding).reshape(N, L, self.num_head, dim // self.num_head).permute(0, 2, 1, 3)  #675*16*1*64-->675*1*16*64
        k_v = self.kv(obs_traj_embedding).reshape(N, -1, 2, self.num_head, dim // self.num_head).permute(2, 0, 3, 1, 4) #675*16*128-->675*16*2*1*64-->2*675*1*16*64
        k, v = k_v[0], k_v[1] #675*1*16*64
        attn = q @ k.transpose(-2, -1)#675*1*16*16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(N, L, dim)#675*1*16*64-->675*16*64
        x = self.proj(x)
        x = self.proj_drop(x)#675*16*64
        x = self.norm2(x)
        x = self.mlp(x)
        return x

class Transformer_Decoder(nn.Module):
    def __init__(self, input_dim, mlp_hid_dim, obs_len=16, num_head=1, drop_rate=0.):
        super(Transformer_Decoder, self).__init__()
        self.num_head = num_head
        self.q = nn.Linear(input_dim, input_dim, bias=False)
        self.kv = nn.Linear(input_dim, input_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(p=drop_rate)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(p=drop_rate)
        self.mlp = Mlp(input_dim, mlp_hid_dim, 2)
        self.pos_emb = nn.Parameter(torch.zeros(1, obs_len, input_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        trunc_normal_(self.pos_emb, std=.02)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, noise_input):
        noise_input = noise_input + self.pos_emb
        noise_input = self.pos_drop(noise_input)
        noise_input = self.norm1(noise_input)
        N, L, dim = noise_input.shape
        q = self.q(noise_input).reshape(N, L, self.num_head, dim // self.num_head).permute(0, 2, 1, 3)
        k_v = self.kv(noise_input).reshape(N, -1, 2, self.num_head, dim // self.num_head).permute(2, 0, 3, 1, 4)
        k, v = k_v[0], k_v[1]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(N, L, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x

class Trajectory_Generator(nn.Module):
    def __init__(self, obs_len, embedding_dim, encoder_input_dim, encoder_output_dim,
                 encoder_mlp_dim, encoder_num_head, drop_rate, rel_traj_dim, noise_dim, merge_mlp_dim):
        super(Trajectory_Generator, self).__init__()
        """
        obs_len:20
        embedding_dim:64
        encoder_input_dim:64,和embedding_dim相等
        encoder_output_dim:64
        encoder_mlp_dim:256
        encoder_num_head:1
        drop_rate = 0
        rel_traj_dim = 64
        noise_dim = 8
        merge_mlp_dim =256
        """
        self.obs_len = obs_len
        self.traj_embedding = Traj_embedding(embedding_dim)
        self.rel_embedding = Traj_embedding(rel_traj_dim)
        self.trans_encoder = Transformer_Encoder(encoder_input_dim, encoder_output_dim,
                                                 encoder_mlp_dim, encoder_num_head, obs_len, drop_rate)
        self.merge_mlp = Mlp(encoder_output_dim+rel_traj_dim, merge_mlp_dim, encoder_output_dim-noise_dim, drop_rate=drop_rate)
        self.noise_dim = noise_dim
        self.trans_decoder = Transformer_Decoder(encoder_output_dim, merge_mlp_dim, obs_len)
        #self.social_mlp = Mlp((encoder_output_dim-noise_dim)*obs_len, 64, 1)
        self.social_mlp = nn.Linear((encoder_output_dim-noise_dim)*obs_len, 1)
        self.sigmoid = nn.Sigmoid()
        self.merge_conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

    def add_noise(self, input):
        if self.noise_dim == 0:
            return input
        else:
            shape = (input.size(0), input.size(1),) + (self.noise_dim, )
            noise = torch.randn(*shape).cuda()
            output = torch.cat([input, noise], dim=-1)
            return output


    def forward(self, obs_traj, obs_traj_rel, seq_start_end):

        obs_traj_embedding = self.traj_embedding(obs_traj_rel)
        obs_traj_embedding = torch.transpose(obs_traj_embedding, 0, 1)  # 交换0，1两个维度的数据
        encoder_output = self.trans_encoder(obs_traj_embedding)#675*16*64

        #Group pool moudle
        merge_output = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            ped_num = end - start
            group_embedding = encoder_output[start:end]#groupsize*16*64
            group_obs_traj = obs_traj.transpose(0, 1)[start:end]#groupsize*16*2
            # Repeat embedding-> H1, H2, H1, H2
            group_embedding_r = group_embedding.repeat(ped_num, 1, 1)#(groupsize*groupsize)*16*64
            # Repeat position -> P1, P2, P1, P2
            group_obs_traj_r1 = group_obs_traj.repeat(ped_num, 1, 1)#(groupsize*groupsize)*16*2
            # Repeat position -> P1, P1, P2, P2
            group_obs_traj_r2 = group_obs_traj.repeat(1, ped_num, 1)
            group_obs_traj_r2 = group_obs_traj_r2.view(ped_num*ped_num, self.obs_len, 2)#(groupsize*groupsize)*16*2

            rel_obs_traj = group_obs_traj_r1 - group_obs_traj_r2#(groupsize*groupsize)*16*2
            rel_pos_embedding = self.rel_embedding(rel_obs_traj)

            merge_embedding = torch.cat((group_embedding_r, rel_pos_embedding), 2)
            merge_embedding = self.merge_mlp(merge_embedding)
            merge_embedding = merge_embedding.view(ped_num, ped_num, self.obs_len, -1)

            #方案2 attention
            social_features = []
            for social_feature in merge_embedding:
                social_feature_f = torch.flatten(social_feature, 1, 2)
                weights = self.sigmoid(self.social_mlp(social_feature_f))
                social_feature = weights.unsqueeze(1) * social_feature
                social_feature = torch.sum(social_feature, dim=0)
                social_features.append(social_feature)
            social_features = torch.cat(social_features, dim=0).view(ped_num, self.obs_len, -1)
            merge_output.append(social_features)

            #方案1 maxpooling
            # merge_embedding = merge_embedding.view(ped_num, ped_num, self.obs_len, -1).max(1)[0]#maxpool//后续换成transformer试
            # merge_output.append(merge_embedding)

            #方案3 conv
            # social_features = []
            # for social_feature in merge_embedding:
            #     social_feature_max = social_feature.max(0)[0].unsqueeze(0)
            #     social_feature_avg = torch.mean(social_feature, dim=0).unsqueeze(0)
            #     social_feature = torch.cat([social_feature_max, social_feature_avg], dim=0).unsqueeze(0)
            #     social_feature = self.merge_conv(social_feature)
            #     social_features.append(social_feature)
            # social_features = torch.cat(social_features, dim=0).view(ped_num, self.obs_len, -1)
            # merge_output.append(social_features)

        merge_output = torch.cat(merge_output, dim=0)
        noise_output = self.add_noise(merge_output)
        pred_traj_rel = self.trans_decoder(noise_output).transpose(0, 1)#转换成socialgan的输出

        #pred_traj_1 = torch.cat((obs_traj[-1,:,:].unsqueeze(0),pred_traj[:-1,:,:]), dim=0)
        #pred_traj_rel = pred_traj - pred_traj_1#求相对坐标
        return pred_traj_rel

class Classfier(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(Classfier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(hid_dim)
        self.fc2 = nn.Linear(hid_dim, output_dim)
        #self.act2 = nn.ReLU()
        #self.norm2 = nn.BatchNorm1d(output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.act1(self.norm1(self.fc1(x)))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Trajectory_Discriminator(nn.Module):
    def __init__(self, obs_len, embedding_dim, encoder_input_dim,
                 encoder_output_dim, mlp_hid_dim, num_head, drop_rate):
        """
        obs_len=40
        embedding_dim=64
        encoder_input_dim = 64
        encoder_output_dim=64
        mlp_hid_dim = 256
        num_head=1
        drop_rate=0
        """
        super(Trajectory_Discriminator, self).__init__()
        self.traj_embedding = Traj_embedding(embedding_dim)
        self.trans_encoder = Transformer_Encoder(encoder_input_dim, encoder_output_dim,
                                                 mlp_hid_dim, num_head, obs_len, drop_rate)
        self.real_classfier = Classfier(encoder_output_dim, 8, 1)
    def forward(self, pre_traj, seq_start_end):
        pre_embedding = self.traj_embedding(pre_traj)  #32*675*64
        pre_embedding = torch.transpose(pre_embedding, 0, 1) #675*32*64
        encoder_output = self.trans_encoder(pre_embedding)  #675*32*64  ///后续加pool模块
        encoder_output = encoder_output.mean(dim=1)
        scores = self.real_classfier(encoder_output)

        return scores


if __name__ == "__main__":
    input = torch.randn(20, 5, 2)
    start_end = torch.tensor([[0,3],[3,5]])
    model = Trajectory_Generator(20, 64, 64, 64, 256, 1, 0, 64,8,256)
    output = model(input, start_end)
    dis_model = Trajectory_Discriminator(20, 64, 64, 64, 256, 1, 0)
    pre = dis_model(output,output,output)
    print("finish")
