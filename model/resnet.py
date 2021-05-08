import torch
import torch.nn as nn
from .spatial_transformer import get_transformer
from .encoder_decoder import get_decoder
from .model_util import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            returned_featmaps = [],
            eval_mode=False,
            use_leaky_relu_for_projection=False,
            multi_cropped_input=True,
            small_image=False,
            double_head=False
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.returned_featmaps = returned_featmaps
        self.small_image = small_image
        self.multi_cropped_input = multi_cropped_input
        self.eval_mode = eval_mode
        self.double_head = double_head
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        if self.small_image:
            self.conv1 = nn.Conv2d(
                3, num_out_filters, kernel_size=5, stride=1, padding=2, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
            )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        if not self.small_image:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
            self.projection_head_2 = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
            if self.double_head:
                self.projection_head_2 = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.LeakyReLU() if use_leaky_relu_for_projection else nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )
            if self.double_head:
                self.projection_head_2 = nn.Sequential(
                    nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                    nn.BatchNorm1d(hidden_mlp),
                    nn.LeakyReLU() if use_leaky_relu_for_projection else nn.ReLU(inplace=True),
                    nn.Linear(hidden_mlp, output_dim),
                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.small_image:
            x = self.maxpool(x)

        
        featmaps = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            level = i + 2
            x = layer(x)
            if level in self.returned_featmaps:
                avg_x = nn.functional.adaptive_avg_pool2d(x, (1,1))
                avg_x = avg_x.view(x.size(0), -1)
                featmaps.append(avg_x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x, featmaps
        

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        return x
    
    def forward_head_2(self, x):
        if self.projection_head_2 is not None:
            x = self.projection_head_2(x)
        
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        return x

    def forward(self, inputs):
        if self.multi_cropped_input:
            assert len(self.returned_featmaps) == 0, 'Can not return feature maps from the backbone for the multi-cropped dataset.'
            if not isinstance(inputs, list):
                inputs = [inputs]
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1], 0)
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            return self.forward_head(output)

        else:
            out, featmaps = self.forward_backbone(inputs)
            h_1 = self.forward_head(out)
            to_return = [h_1,]
            
            if self.double_head:
                to_return.append(self.forward_head_2(out))
            
            if len(featmaps):
                to_return.append(featmaps)
            
            return to_return


class STN_Resnet_VAE(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            eval_mode=False,
            small_image=True,
            input_shape=[3,96,96],
            stn_latent_size=64,
            vae_latent_size=128,
            encoder_hidden_size = 1024,
            penalize_view_similarity=True
    ):
        super(STN_Resnet_VAE, self).__init__()
        self.penalize_view_similarity = penalize_view_similarity
        self.stn = get_transformer('affine_RNN')(input_shape, stn_latent_size)
        self.resnet = ResNet(
            block,
            layers,
            zero_init_residual,
            groups,
            widen,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
            normalize,
            hidden_mlp=encoder_hidden_size,
            output_dim=vae_latent_size,
            returned_featmaps = [],
            eval_mode=False,
            use_leaky_relu_for_projection=True,
            multi_cropped_input=False,
            double_head=True
        )
        self.stn_latent_size = stn_latent_size
        self.vae_latent_size = vae_latent_size
        self.decoder_output_shape = calculate_new_shape(input_shape, [1,1/4,1/4])
        self.decoder = get_decoder('conv')(self.decoder_output_shape, vae_latent_size + 6)
        '''
        self.vae_input_size = resnet_output_size
        self.vae = get_VAE('VanillaVAE_MLP')(
                self.vae_input_size, 
                vae_latent_size, 
                encoder_type='mlp',
                decoder_type='conv',
                with_decoder=self.with_decoder
        )
        '''
        # When contrasing latent representations/reconstructions,
        # since the likelihood and prior is modelled with gaussian, 
        # we end up with MSE after omitting variance and other constants.
        self.contrastive_loss = nn.MSELoss()
        self.view_similarity = nn.CosineSimilarity(dim=1)

        self.visuals = {}

    def get_current_visuals(self,):
        return self.visuals

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs):
        # first, extract views with RNN-STN randomly, except that the second conditioned on the first
        # initial hidden state of the RNN-STN is sampled from normal dist.
        h_0 = torch.normal(0, 1, (inputs.size(0), self.stn_latent_size)).to(device)
        views, thetas = self.stn(inputs, h_0)

        mus = []
        log_vars = []
        recons = []
        recons_target = []
        # loop through the views. there should be two of them.
        for (view, theta) in zip(views, thetas[::-1]):
            # extract features with resnet
            mu, log_var, = self.resnet(view)
            mus.append(mu)
            log_vars.append(log_var)
            
            # each feature-map is combined with affine transf. params of the other (theta list is reversed)
            latent = self.reparameterize(mu, log_var)
            conditioned_latent = torch.cat([latent, theta], dim=1)
            recon = self.decoder(conditioned_latent)
            recons.append(recon)
            
            # downsample the input for simplified recons. target
            target = nn.functional.interpolate(view, scale_factor=1/4)
            recons_target.append(target)
        
        # cache the first four of the inputs and extracted views for visualisation
        self.visuals = {'img':inputs[:4].detach().cpu(), 
                        'view_1':views[0][:4].detach().cpu(), 
                        'view_2':views[1][:4].detach().cpu()
                        }
        return {'recons':recons, 'mu':mus, 'log_var':log_vars, 'theta':thetas, 'input':recons_target[::-1]}


    def KLD(self, mu, log_var):
        '''
           Kulback-Leibler Divergence for calculating the divergence between latent vars and normal dist.
        '''
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)


    def calculate_loss(self, outputs):
        '''
            Combine the losses from VAE objective and Consrastive objective.
        '''
        total_loss = 0.
        loss_vars = {}
        # loop trough the views extracted with RNN-SPN. There should be two of them. 
        for idx, recon in enumerate(outputs['recons']):
            # add reconstruction loss to total loss, for the current crop/view in the outputs. 
            recons_loss = self.contrastive_loss(recon, outputs['input'][idx])
            loss_vars['reconstruction_loss'] = recons_loss
            total_loss = total_loss + recons_loss
            
            # add total VAE loss (KLD) to total loss, for the current crop/view in the outputs. 
            kld_loss = self.KLD(outputs['mu'][idx], outputs['log_var'][idx])
            loss_vars['KLD_' + str(idx)] = kld_loss            
            total_loss = total_loss + kld_loss

        loss_vars['view_similarity_loss'] = 0.
        if self.penalize_view_similarity:
            # penalize too similar views (cosine(v1, v2) > 0.9)
            view_sim = self.view_similarity(outputs['theta'][0], outputs['theta'][1])
            zero_tensor = torch.FloatTensor([0]).to(outputs['theta'][0].device)
            view_sim_loss = torch.max(zero_tensor, view_sim - 0.9).mean()
            loss_vars['view_similarity_loss'] = view_sim_loss
            total_loss = total_loss + view_sim_loss

        return total_loss, loss_vars 
        

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet50w2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)

def resnet50w4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)

def resnet50w5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)

def stn_resnet18_vae(**kwargs):
    return STN_Resnet_VAE(BasicBlock, [2, 2, 2, 2], **kwargs)

def stn_resnet34_vae(**kwargs):
    return STN_Resnet_VAE(BasicBlock, [3, 4, 6, 3], **kwargs)

