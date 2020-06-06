from dataclasses import dataclass


@dataclass
class VirtualGPU:
    """
    gpu_memory_limit: Max memory in MB for  virtual device. Setting LOWER than total available this can help
                      avoid out of memory errors on some set ups when TF tries to allocate too much memory
                      (seems to be a bug).
    gpu_device_id: Integer device identifier for the real GPU the virtual GPU should use.
    """
    gpu_memory_limit: int = 512
    gpu_device_id: int = 0

    def __post_init__(self):
        self.on = self._set_tf()

    def _set_tf(self) -> bool:
        """
        Helper function for training on tf. Reduces GPU memory footprint for keras/tf models.

        Creates a virtual device on the request GPU with limited memory. Will fail gracefully if GPU isn't available.

        :return: Bool indicating if TF appears to be running on GPU. Can be used, for example, to avoid using
                 multiprocessing in the caller when running on GPU. This will likely result in an exception, but may
                 result in hanging forever, so probably best avoided.
        """
        import tensorflow as tf

        gpu = True
        try:
            # Handle running on GPU: If available, reduce memory commitment to avoid over-committing error in 2.2.0 and
            # for also for general convenience.
            tf.config.experimental.set_virtual_device_configuration(
                tf.config.experimental.list_physical_devices('GPU')[self.gpu_device_id],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=self.gpu_memory_limit)])
        except AttributeError:
            # Assuming not using GPU
            gpu = False
        except IndexError:
            # Assuming using GPU but indexed device not found.
            gpu = False

        return gpu
