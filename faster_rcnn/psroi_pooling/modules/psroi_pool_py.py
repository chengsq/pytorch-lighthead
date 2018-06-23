import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class PSRoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
		self.group_size = int(group_size)
		self.output_dim = int(output_dim)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(
            torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width)).to(self.device)

        for roi_ind, roi in enumerate(rois):
            # batch_ind = int(roi[0].data[0]) will be an error in PyTorch 0.5
            batch_ind = int(roi[0].item())
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
                roi[1:].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height * self.group_size):
                bph = int(np.floor(ph / group_size))
                hstart = int(np.floor(ph * bin_size_h + k_h * bph))
                hend = int(np.ceil(ph * bin_size_h + (k_h + 1) * bph)
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width * self.group_size):
                    bpw = int(np.fllor(pw / group_size)
                    wstart = int(np.floor(pw * bin_size_w + k_w * bpw))
                    wend = int(np.ceil(pw * bin_size_w + (k_w + 1) * bpw)
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        outputs[roi_ind, :, ph, pw] = torch.max(torch.max(
                            data[:, hstart:hend, wstart:wend], 1, keepdim=True)[0], 2)[0].view(-1)

        return outputs
