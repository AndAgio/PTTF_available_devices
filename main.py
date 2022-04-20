import tensorflow as tf
import torch


def get_devices_tf():
    physical_devices = tf.config.list_physical_devices()
    print('Detected physical devices: {}'.format(physical_devices))
    cpus = [phy.name for phy in physical_devices if phy.device_type == 'CPU']
    print('Detected CPUs: {}'.format(cpus))
    gpus = [phy.name for phy in physical_devices if phy.device_type == 'GPU']
    print('Detected GPUs: {}'.format(gpus))
    return gpus


def check_gpus_usage_tf(gpus):
    if len(gpus) == 0:
        raise ValueError('Found empty list of CUDA devices...')
    else:
        for gpu in gpus:
            gpu_name = gpu.replace('physical_device', 'device')
            with tf.device(gpu_name):
                try:
                    tf.zeros((100, 100), dtype=tf.dtypes.float32)
                    print('Device {} is ok'.format(gpu))
                except:
                    print('Device {} didn\'t pass initialization test!!!'.format(gpu))


def get_devices_torch():
    gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print('Detected GPUs: {}'.format(gpus))
    return gpus


def check_gpus_usage_torch(gpus):
    if len(gpus) == 0:
        raise ValueError('Found empty list of CUDA devices...')
    else:
        for gpu in gpus:
            try:
                torch.zeros((100, 100), dtype=torch.float32, device=gpu)
                print('Device {} is ok'.format(gpu))
            except:
                print('Device {} didn\'t pass initialization test!!!'.format(gpu))


def main():
    print('\n=========================================================')
    print('======================= TENSORFLOW ======================')
    print('=========================================================')
    gpus = get_devices_tf()
    check_gpus_usage_tf(gpus)
    print('\n=========================================================')
    print('========================= PYTORCH =======================')
    print('=========================================================')
    gpus = get_devices_torch()
    check_gpus_usage_torch(gpus)


if __name__ == '__main__':
    main()