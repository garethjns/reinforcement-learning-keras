import cv2
import gym
import numpy as np


class ImageProcessWrapper(gym.ObservationWrapper):
    """
    Convert frames from (210, 160, 3) or (250, 160, 3) to (84, 84, 1).
    Scales from 0 -> 255 to 0 -> 1.

    Based on ProcessFrame84, ScaledFloatFrame, ImageToPyTorch wrappers:
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # New env obs space shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, 1),
                                                dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return ImageProcessWrapper.process(obs)

    @staticmethod
    def process(frame: np.ndarray) -> np.ndarray:
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            raise ValueError("Unknown resolution.")
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1]).astype(np.float32) / 255.0

        return x_t.squeeze()
