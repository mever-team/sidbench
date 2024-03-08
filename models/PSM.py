import torch
import torch.nn as nn
from networks.resnet_psm import resnet50
import torch.nn.functional as F


class SALayer(nn.Module):

    def __init__(self, dim=128, head_size=4):
        super(SALayer, self).__init__()
        self.mha=nn.MultiheadAttention(dim, head_size)
        self.ln1=nn.LayerNorm(dim)
        self.fc1=nn.Linear(dim, dim)
        self.ac=nn.ReLU()
        self.fc2=nn.Linear(dim, dim)
        self.ln2=nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, len_size, fea_dim=x.shape
        x=torch.transpose(x,1,0)
        y,_=self.mha(x,x,x)
        x=self.ln1(x+y)
        x=torch.transpose(x,1,0)
        x=x.reshape(batch_size*len_size, fea_dim)
        x=x+self.fc2(self.ac(self.fc1(x)))
        x=x.reshape(batch_size,len_size, fea_dim)
        x=self.ln2(x)

        return x


class COOI(): # Coordinates On Original Image
    def __init__(self):
        self.stride=32
        self.cropped_size=224
        self.score_filter_size_list=[[3,3],[2,2]]
        self.score_filter_num_list=[3,3]
        self.score_nms_size_list=[[3,3],[3,3]]
        self.score_nms_padding_list=[[1,1],[1,1]]
        self.score_corresponding_patch_size_list=[[224, 224], [112, 112]]
        self.score_filter_type_size=len(self.score_filter_size_list)

    def get_coordinates(self, fm, scale):
        with torch.no_grad():
            batch_size, _, fm_height, fm_width=fm.size()
            scale_min=torch.min(scale, axis=1, keepdim=True)[0].long()
            scale_base=(scale-scale_min).long()//2 # torch.div(scale-scale_min,2,rounding_mode='floor')
            input_loc_list=[]
            for type_no in range(self.score_filter_type_size):
                score_avg=nn.functional.avg_pool2d(fm, self.score_filter_size_list[type_no], stride=1) #(7,2048,5,5), (7,2048,6,6)
                score_sum=torch.sum(score_avg, dim=1, keepdim=True) #(7,1,5,5), (7,1,6,6) # since the last operation in layer 4 of the resnet50 is relu, thus the score_sum are greater than zero
                _,_,score_height,score_width=score_sum.size()
                patch_height, patch_width=self.score_corresponding_patch_size_list[type_no]

                for filter_no in range(self.score_filter_num_list[type_no]):
                    score_sum_flat=score_sum.view(batch_size, -1)
                    value_max,loc_max_flat=torch.max(score_sum_flat, dim=1)
                    #loc_max=torch.stack((torch.div(loc_max_flat,score_width,rounding_mode='floor'), loc_max_flat%score_width), dim=1)
                    loc_max=torch.stack((loc_max_flat//score_width, loc_max_flat%score_width), dim=1)
                    top_patch=nn.functional.max_pool2d(score_sum, self.score_nms_size_list[type_no], stride=1, padding=self.score_nms_padding_list[type_no])
                    value_max=value_max.view(-1,1,1,1)
                    erase=(top_patch!=value_max).float() # due to relu operation, the value are greater than 0, thus can be erase by multiply by 1.0/0.0
                    score_sum=score_sum*erase

                    # location in the original images
                    loc_rate_h=(2*loc_max[:,0]+fm_height-score_height+1)/(2*fm_height)
                    loc_rate_w=(2*loc_max[:,1]+fm_width-score_width+1)/(2*fm_width)
                    loc_rate=torch.stack((loc_rate_h, loc_rate_w), dim=1)
                    loc_center=(scale_base+scale_min*loc_rate).long()
                    loc_top=loc_center[:,0]-patch_height//2
                    loc_bot=loc_center[:,0]+patch_height//2+patch_height%2
                    loc_lef=loc_center[:,1]-patch_width//2
                    loc_rig=loc_center[:,1]+patch_width//2+patch_width%2
                    loc_tl=torch.stack((loc_top, loc_lef), dim=1)
                    loc_br=torch.stack((loc_bot, loc_rig), dim=1)

                    # For boundary conditions
                    loc_below=loc_tl.detach().clone() # too low
                    loc_below[loc_below>0]=0
                    loc_br-=loc_below
                    loc_tl-=loc_below
                    loc_over=loc_br-scale.long() # too high
                    loc_over[loc_over<0]=0
                    loc_tl-=loc_over
                    loc_br-=loc_over
                    loc_tl[loc_tl<0]=0 # patch too large

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

            input_loc_tensor=torch.stack(input_loc_list, dim=1) # (7,6,4)
            #print(input_loc_tensor)
            return input_loc_tensor


class PSM(nn.Module):
    def __init__(self):
        super(PSM, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.COOI=COOI()
        self.mha_list=nn.Sequential(
            SALayer(128, 4),
            SALayer(128, 4),
            SALayer(128, 4)
        )

        self.fc1=nn.Linear(2048, 128)
        self.ac=nn.ReLU()
        self.fc=nn.Linear(128, 1)

    def forward(self, input_img, cropped_img, scale):
        x = cropped_img
        batch_size, p, _, _ = x.shape # [batch_size, 3, 224, 224]

        fm, whole_embedding = self.resnet(x) # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]

        s_whole_embedding = self.ac(self.fc1(whole_embedding)) #128
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)

        input_loc = self.COOI.get_coordinates(fm.detach(), scale) 

        _,proposal_size,_ = input_loc.size()
        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(fm.device)  # [N, 4, 3, 224, 224]
        
        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t,l,b,r = input_loc[batch_no, proposal_no]
                img_patch = input_img[batch_no][:, t:b, l:r]
                _, patch_height, patch_width=img_patch.size()
                if patch_height == 224 and patch_width == 224:
                    window_imgs[batch_no, proposal_no] = img_patch
                else:
                    window_imgs[batch_no, proposal_no:proposal_no+1] = F.interpolate(img_patch[None,...], size=(224, 224), mode='bilinear', align_corners=True)  # [N, 4, 3, 224, 224]

        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 224, 224)  # [N*4, 3, 224, 224] 
        _, window_embeddings = self.resnet(window_imgs.detach()) # [batchsize*self.proposalN, 2048]
        s_window_embedding = self.ac(self.fc1(window_embeddings)) # [batchsize*self.proposalN, 128]
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 128)

        all_embeddings = torch.cat((s_window_embedding, s_whole_embedding), 1) # [1, 1+self.proposalN, 128]
        
        all_embeddings = self.mha_list(all_embeddings)
        all_logits = self.fc(all_embeddings[:,-1])
        
        return all_logits

    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        try:
            self.load_state_dict(state_dict['model'], strict=True)
        except:
            self.load_state_dict(state_dict)

    def predict(self, input_img, cropped_img, scale):
        with torch.no_grad():
            logits = self.forward(input_img, cropped_img, scale)
            return logits.sigmoid().flatten().tolist()
        