import torch.nn as nn


def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)
