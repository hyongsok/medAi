import sys
import platform
try:
    import torch
except:
    print('Could not load torch. Check your installation.')
    sys.exit(-1)

try:
    import torchvision
except:
    torchvision = None

if __name__ == '__main__':
    print('pytorch version', torch.__version__)
    if torchvision is None:
        print('torchvision not found.')
    else:
        print('torchvision version', torchvision.__version__, '\n')
    print('CPU:', platform.processor(),'\n')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for ii in range(num_gpus):
            prop = torch.cuda.get_device_capability(ii)
            print('CUDA device {ii}: {prop.name}\nMemory: {prop.total_memory}\nMulti processor count: {prop.multi_processor_count}\nMajor version: {prop.major}, minor version {prop.minor}\n'.format(ii=ii, prop=prop))
    else:
        print('No CUDA devices')