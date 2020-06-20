from dataclasses import dataclass
import warnings


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
        self.physical_device = None
        self.virtual_device = None
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

        try:
            self.physical_device = tf.config.experimental.list_physical_devices('GPU')[self.gpu_device_id]
            virtual_device = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=self.gpu_memory_limit)
        except (IndexError, AttributeError) as e:
            # IndexError: Assuming using GPU but indexed device not found.
            # AAttributeError: Assuming no GPU.
            warnings.warn(f"Not using GPU due to: {e}")
            return False

        # First check a virtual device hasn't already been set. If it has, we don't want to try and set a new one.
        # - If the device has not been used before, it will be replaced and no error is raised
        # - If the device has been used before it will be initialised and will be immutable, raising a RuntimeError
        #   on set_virtual_device_configuration call.
        # Aim here is to make behaviour more predictable. This allows multiple models to run in the same session, as
        # long as the max memory required is set at the start. The alternative is it fixing to the first used model's
        # memory requirement (for example in a session that calls VirtualGPU again in agent.example()).
        existing_device = tf.config.experimental.get_virtual_device_configuration(self.physical_device)
        if existing_device is not None:
            warnings.warn(f"A virtual GPU with {existing_device[0].memory_limit} MB memory already exists, "
                          f"using this rather than creating another with the requested {self.gpu_memory_limit} MB."
                          f"Good luck.")
            self.virtual_device = existing_device[0]
            self.gpu_memory_limit = existing_device[0].memory_limit

        else:
            # GPU available and no existing virtual device. Create a new one.
            tf.config.experimental.set_virtual_device_configuration(self.physical_device, [virtual_device])

        return True
