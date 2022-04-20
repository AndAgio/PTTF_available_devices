import tensorflow as tf


def get_devices_tf():
    physical_devices = tf.config.list_physical_devices()
    print('Detected physical devices: {}'.format(physical_devices))
    cpus = [phy for phy in physical_devices if phy.device_type == 'CPU']
    print('Detected CPUs: {}'.format(cpus))
    gpus = [phy for phy in physical_devices if phy.device_type == 'GPU']
    print('Detected GPUs: {}'.format(gpus))
    return gpus


def check_gpus_usage_tf(gpus):
    if len(gpus) == 0:
        raise ValueError('Found empty list of CUDA devices...')
    else:
        for i, gpu in enumerate(gpus):
            gpu_name = gpu.name.replace('physical_device', 'device')
            details = tf.config.experimental.get_device_details(gpu)
            device_name = details.get('device_name', 'Unknown GPU')
            with tf.device(gpu_name):
                try:
                    tf.zeros((2, 2), dtype=tf.dtypes.float32)
                    print('Device {} is ok'.format(device_name))
                except:
                    print('Device {} didn\'t pass initialization test!!!'.format(device_name))


def main():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('\n=========================================================')
    print('======================= TENSORFLOW ======================')
    print('=========================================================')
    gpus = get_devices_tf()
    check_gpus_usage_tf(gpus)


if __name__ == '__main__':
    main()
