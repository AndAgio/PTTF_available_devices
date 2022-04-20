import torch


def get_devices_torch():
    gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print('Detected GPUs: {}'.format(gpus))
    return gpus


def check_gpus_usage_torch(gpus):
    if len(gpus) == 0:
        raise ValueError('Found empty list of CUDA devices...')
    else:
        for i, _ in enumerate(gpus):
            print(torch.cuda.get_device_name(i))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
            with torch.cuda.device(i):
                try:
                    torch.zeros((2, 2), dtype=torch.float32, device='cuda')
                    print('Device {} is ok'.format(torch.cuda.get_device_name(i)))
                except:
                    print('Device {} didn\'t pass initialization test!!!'.format(torch.cuda.get_device_name(i)))
                torch.zeros((2, 2), dtype=torch.float32, device='cuda')


def main():
    print('\n=========================================================')
    print('========================= PYTORCH =======================')
    print('=========================================================')
    gpus = get_devices_torch()
    check_gpus_usage_torch(gpus)


if __name__ == '__main__':
    main()
