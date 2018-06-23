int psroi_pooling_forward(int pooled_height, int pooled_width, float spatial_scale,
                        int group_size, int output_dim, 
                        THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output);
