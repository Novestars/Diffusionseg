import einops
import nibabel
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from einops import rearrange
import numpy as np
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss
from positional_encodings.torch_encodings import PositionalEncoding2D
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,stride = 1,relu='lrelu'):
        super().__init__()
        if relu=='lrelu':
            self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.01)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1,stride=stride)
        self.bn1 = nn.InstanceNorm3d(middle_channels,affine=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_channels,affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class SelfAttentionLayer2(nn.Module):

    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.norm = nn.LayerNorm(d_model)
        self.num_heads = nhead
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    def attn(self,q, k,v):

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        tgt_local = (attn @ v).transpose(1, 2)
        return tgt_local
    def forward(self, tgt,ind):
        #token_num = tgt.shape[1]
        tgt_local = self.norm(tgt)
        if ind %2==1:
            tgt_local = einops.rearrange(tgt_local, 'b (h p1 w p2 d p3) c-> (b p1 p2 p3) (h w d) c',p1=4,p2=4,p3=4,h=4,w=4,d=4)
        else:
            tgt_local = einops.rearrange(tgt_local, 'b (h p1 w p2 d p3) c-> (b h w d) (p1 p2 p3) c',p1=4,p2=4,p3=4,h=4,w=4,d=4)
        B, N, C = tgt_local.shape
        qkv = self.qkv(tgt_local).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        #tgt_local = self.attn(q,k,v)
        tgt_local = torch.utils.checkpoint.checkpoint(self.attn,q,k,v)
        tgt_local = self.proj(tgt_local.reshape(B, N, C))
        if ind %2==1:
            tgt_local = einops.rearrange(tgt_local, ' (b p1 p2 p3) (h w d) c ->b (h p1 w p2 d p3) c', p1=4,p2=4,p3=4,h=4,w=4,d=4)
        else:
            tgt_local = einops.rearrange(tgt_local, '(b h w d) (p1 p2 p3) c->b (h p1 w p2 d p3) c',p1=4,p2=4,p3=4,h=4,w=4,d=4)
        y = tgt + tgt_local
        return y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        #self.attn = ChannelAttention(dim)
    def forward(self, x):
        y = self.norm(x)
        y = self.net(y)
        y =y +x
        return y

class UNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        nb_filter = [32, 64, 256, 1024]
        self.nb_filter = nb_filter
        self.up = nn.Upsample(scale_factor=(2,2,2), mode='nearest')

        self.conv0_01 = VGGBlock(2, nb_filter[0], nb_filter[0])
        self.conv1_01 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],stride=2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],stride=2)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],stride=2)

        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_11 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.unet_attention = nn.ModuleList()
        self.unet_ffn = nn.ModuleList()
        self.unet_attention_num =6
        for i in range(0, self.unet_attention_num):
                self.unet_attention.append(SelfAttentionLayer2(nb_filter[-1],8))
                self.unet_ffn.append(FeedForward(nb_filter[-1], nb_filter[-1]*4))
        self.out_norm = nn.LayerNorm(nb_filter[-1])
        self.final1 = nn.Sequential(    nn.Conv3d(32,2,kernel_size=1,bias=True))

    def forward(self, input, seg):
        ################################################################################################################
        x = torch.cat([input,seg],1)

        x0_0 = self.conv0_01(x)
        x1_0 = self.conv1_01(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)

        b,c,h,w,d = x3_0.shape
        x3_0_faltten = rearrange(x3_0,'b c h w d-> b (h w d) c')
        for i in range(self.unet_attention_num):
            x3_0_faltten = self.unet_attention[i](x3_0_faltten,i)
            x3_0_faltten = self.unet_ffn[i](x3_0_faltten)
        x3_0_faltten = self.out_norm(x3_0_faltten)

        x3_1 = rearrange(x3_0_faltten,'b (h w d) c->b c h w d',h=h,w=w,d=d)

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_11(torch.cat([x0_0, self.up(x1_1)], 1))
        out = self.final1(x0_1)
        return out


def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


class NP(LightningModule):
    def __init__(
            self,
            lr,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = UNet()

        # self.l1 = nn.SmoothL1Loss()
        # self.dice = nn.SmoothL1Loss()
        self.dice_ce = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {},weight_dice=0)
        self.dc = SoftDiceLoss(apply_nonlin=None, batch_dice=True, do_bg=True, smooth=1e-5)
        self.CE = nn.CrossEntropyLoss()
        '''self.label_list = np.array([   2.,   3.,   4.,   5.,   7.,   8.,  10.,  11.,  12.,  13.,  14.,
         15.,  16.,  17.,  18.,  24.,  26.,  28.,  30.,  31.,  41.,  42.,  43.,
         44.,  46.,  47.,  49.,  50.,  51.,  52.,  53.,  54.,  58.,  60.,  62.,
         63.,  72.,  77.,  80.,  85., 251., 252., 253., 254., 255.]).astype(int)'''
        self.label_list = np.array([2., 3., 4., 41., 42.]).astype(int)
        self.label_list = np.array([2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 43
                                       , 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 72, 77, 80, 85, 251, 252,
                                    253, 254, 255, 11101, 11102, 11103, 11104, 11105, 11106, 11107,
                                    11108, 11109, 11110, 11111, 11112, 11113, 11114, 11115, 11116, 11117, 11118,
                                    11119, 11120, 11121, 11122, 11123, 11124, 11125, 11126, 11127, 11128, 11129,
                                    11130, 11131, 11132, 11133, 11134, 11135, 11136, 11137, 11138, 11139, 11140, 11141,
                                    11143, 11144, 11145, 11146, 11147, 11148, 11149, 11150, 11151, 11152, 11153, 11154,
                                    11155,
                                    11156, 11157, 11158, 11159, 11160, 11161, 11162, 11163, 11164, 11165, 11166, 11167,
                                    11168,
                                    11169, 11170, 11171, 11172, 11173, 11174, 11175, 12101, 12102, 12103, 12104, 12105,
                                    12106, 12107,
                                    12108, 12109, 12110, 12111, 12112, 12113, 12114, 12115, 12116, 12117, 12118, 12119,
                                    12120, 12121,
                                    12122, 12123, 12124, 12125, 12126, 12127, 12128, 12129, 12130, 12131, 12132, 12133,
                                    12134, 12135,
                                    12136, 12137, 12138, 12139, 12140, 12141, 12143, 12144, 12145, 12146, 12147, 12148,
                                    12149, 12150,
                                    12151, 12152, 12153, 12154, 12155, 12156, 12157, 12158, 12159, 12160, 12161, 12162,
                                    12163, 12164,
                                    12165, 12166, 12167, 12168, 12169, 12170, 12171, 12172, 12173, 12174, 12175])
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        imgs, gts, brain_mask, ind = batch

        selected_gts = gts ==2
        selected_gts = selected_gts.type_as(imgs)

        #threshold = np.random.uniform(0,1)
        t = np.random.randint(0, 6,5)
        threshold_list = [0.01,0.02,0.05,0.1,0.2,0.5]
        threshold = [threshold_list[i] for i in t]
        noise = torch.rand_like(selected_gts) < torch.Tensor(threshold).reshape(5,1,1,1,1).type_as(imgs)
        selected_gts[noise] = 1 - selected_gts[noise]

        #self.generated_imgs = self.generator(imgs, selected_gts)

        #self.generated_imgs = self.generator(imgs, selected_gts)
        if np.random.uniform(0,1)>0.5:
            self.generated_imgs = self.generator(imgs, selected_gts)
        else:
            self.generated_imgs = self.generator(imgs, selected_gts).detach()
            output = nn.functional.softmax(self.generated_imgs, 1)
            pred = torch.argmax(output, dim=1, keepdim=True)
            selected_gts[pred.bool()] = 1 - selected_gts[pred.bool()]
            noise[pred.bool()] = ~ noise[pred.bool()]
 
            self.generated_imgs = self.generator(imgs, selected_gts)

        #pred = nn.functional.softmax(self.generated_imgs,1)

        loss = self.dice_ce(self.generated_imgs, noise)
        # train generator
        # generate images
        '''
                    output = nn.functional.softmax(self.generated_imgs, 1)
            pred = torch.argmax(output, dim=1, keepdim=True)
            selected_gts[pred.bool()] = 1 - selected_gts[pred.bool()]

        '''
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        if self.trainer.is_last_batch:
            sch.step()

        self.log("l1_train", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        # my_list = ['final_feat','conv0_1','conv1_1','conv2_1','conv3_1','conv4_1','conv0_0','conv1_0','conv2_0','conv3_0','conv4_0','seg_conv0_0','seg_conv1_0','seg_conv2_0','seg_conv3_0','seg_conv4_0']
        my_list = []

        sparse_params = list(filter(lambda kv: kv[0].split('.')[0] in my_list, self.generator.named_parameters()))
        sparse_params = [i[1] for i in sparse_params]

        base_params = list(filter(lambda kv: kv[0].split('.')[0] not in my_list, self.generator.named_parameters()))
        base_params = [i[1] for i in base_params]
        # sparse_params = [{"params": sparse_params,'lr': lr,'weight_dacay': 5e-2},]
        sparse_params = [{"params": sparse_params, 'lr': lr, 'weight_decay': 1e-4}]
        base_params = [{"params": base_params, 'weight_decay': 5e-2,'lr': 1e-4/2}, ]

        optimizer = torch.optim.AdamW(base_params, lr=lr)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1/200,total_iters=200,verbose=True)

        return [optimizer], [lr_scheduler, ]

    def validation_step(self, batch, batch_idx):
        imgs, gts, brain_mask, ind = batch

        #selected_gts = torch.logical_or(gts == 2, (gts == 41)).float()
        #selected_gts = torch.logical_or(gts == 4, (gts == 7)).float()
        selected_gts = gts ==2
        selected_gts = selected_gts.type_as(imgs)

        #noise = torch.rand_like(selected_gts) > 0.5
        threshold = 0.5

        noise = torch.rand_like(selected_gts) < threshold
        selected_gts[noise] = 1 - selected_gts[noise]

        #noise = noise.type_as(imgs)
        for i in range(10):
            self.generated_imgs = self.generator(imgs, selected_gts)
            output = nn.functional.softmax(self.generated_imgs, 1)
            pred = torch.argmax(output, dim=1, keepdim=True)
            selected_gts[pred.bool()] = 1 - selected_gts[pred.bool()]
            print(dice_coef(gts ==2, selected_gts))
            pass
        dice = dice_coef(gts ==2, selected_gts)

        self.log("dice_val", dice, prog_bar=True, sync_dist=True)

        return dice


'''
nvidia-smi | grep 'g11' | awk '{ print $5 }' | xargs -n1 kill -9
'''








