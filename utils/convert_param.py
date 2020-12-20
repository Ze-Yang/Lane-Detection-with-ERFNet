# code for converting the pretrained weights from pytorch to paddle.
# Remember to install torch and uncomment the following line if you need
# to use this script.

# import torch
import paddle.fluid as fluid
from models.erfnet_pad import ERFNet

if __name__ == '__main__':
    weight = 'pretrained/ERFNet_pretrained.tar'
    num_class = 20
    img_height, img_width = 384, 1024
    with fluid.dygraph.guard():
        model = ERFNet(num_class, [img_height, img_width])
        checkpoint = torch.load(weight, map_location='cpu')
        cnt = 0
        stdict = {}

        for name, param in checkpoint['state_dict'].items():
            name = name.replace('module.', '')
            if 'running_mean' in name:
                name_pad = name.replace('running_mean', '_mean')
            elif 'running_var' in name:
                name_pad = name.replace('running_var', '_variance')
            else:
                name_pad = name
            if name_pad in model.state_dict().keys():
                shape_pad = model.state_dict()[name_pad].shape
                shape_torch = list(checkpoint['state_dict']['module.' + name].shape)
                if str(shape_pad) == str(shape_torch):
                    stdict[name_pad] = checkpoint['state_dict']['module.' + name].cpu().numpy()
                    cnt += 1
                else:
                    print(name)
                    stdict[name_pad] = model.state_dict()[name_pad].numpy()
            else:
                print(name)
        model.load_dict(stdict)
        fluid.dygraph.save_dygraph(model.state_dict(), 'pretrained/ERFNet_pretrained')
        print(cnt)
