"""
A simple Farama Foundation gymasium environment consisting of a white dot moving in a black
square.
"""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pygame


class ALE(object):
    """Class to support atari_wrappers in OpenAI baselines."""
    def __init__(self):
        self.lives = lambda: 0


class MovingDotEnv(gym.Env):
    """ 
    Base class for MovingDot game 

    Args:
        render_mode: chose environemnt render mode - human or rgb_array
        random_start: if True, dot randomly starts on canvas, or in top left if False
        max_steps: maximum number of steps in an episode before truncation occurs
    """
    metadata = {'render_modes': ['human','rgb_array'], 'render_fps': 10}

    def __init__(self, render_mode='human', random_start=True, max_steps=1000):
        """Initialize parent dot environment."""
        super(gym.Env, self).__init__()

        # Environment parameters
        self.random_start = random_start
        self.max_steps = max_steps
        self.dot_size = (4, 4)
        self.window_width = 210
        self.window_height = 160

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # environment setup
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.window_height, self.window_width, 3),
                                            dtype=np.uint8)
        self.center = np.array([int(self.window_width / 2), int(self.window_height / 2)])
        
        # render setup
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # atari_wrapper compatability
        self.ale = ALE()


    def reset(self, options={}, seed=None):
        """Reset the environment."""
        super().reset(seed=seed, options=options) # handles the np_random
        # self.np_random, _ = seeding.np_random(seed)

        if self.random_start:
            x = self.np_random.integers(low=0, high=self.window_width)
            y = self.np_random.integers(low=0, high=self.window_height)
            self.pos = (x, y)
        else:
            self.pos = (self.dot_size[0], self.dot_size[1])
        self.steps = 0
        ob = self._get_ob()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return ob, info

    def step(self, action):
        prev_pos = self.pos[:]

        self._update_pos(action)

        ob = self._get_ob()
        info = self._get_info()

        self.steps += 1
        if self.steps < self.max_steps:
            truncated = False
        else:
            truncated = True
        terminated = False # No built-in early stop

        dist1 = np.linalg.norm(prev_pos - self.center)
        dist2 = np.linalg.norm(self.pos - self.center)
        if dist2 < dist1:
            reward = 1
        elif dist2 == dist1:
            reward = 0
        else:
            reward = -1

        if self.render_mode == "human":
            self._render_frame()

        return ob, reward, truncated, terminated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        """Render frame to return to user."""
        # for human render mode, change window and clock to not None
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # create empty map
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 0, 0))
        # place dot
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                self.pos,
                self.dot_size,
            ),
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    
    def get_action_meanings(self):
        return ['NOOP', 'DOWN', 'RIGHT', 'UP', 'LEFT']
    
    def _update_pos(self, action):
        """ subclass is supposed to implement the logic
            to update the frame given an action at t. """
        raise NotImplementedError
    
    def _get_ob(self):
        """Return environment reflecting dot's position."""
        ob = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        x = self.pos[0]
        y = self.pos[1]
        w = self.dot_size[0]
        h = self.dot_size[1]
        ob[y - h:y + h, x - w:x + w, :] = 255
        return ob
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self.pos - self.center)}

class MovingDotDiscreteEnv(MovingDotEnv):
    """ 
    Discrete Action MovingDot env 
    
    Args:
        step_size: how many pixels the dot moves per action
        -- Parent Class Arguments --
        render_mode: chose environemnt render mode - human or rgb_array
        random_start: if True, dot randomly starts on canvas, or in top left if False
        max_steps: maximum number of steps in an episode before truncation occurs
    """
    def __init__(self, step_size=1, **kwargs):
        super(MovingDotDiscreteEnv, self).__init__(**kwargs)
        self.action_space = spaces.Discrete(5)
        self.step_size = step_size

    def get_action_meanings(self):
        return ['NOOP', 'DOWN', 'RIGHT', 'UP', 'LEFT']

    def _update_pos(self, action):
        assert action >= 0 and action <= 4
        _x, _y = self.pos
        if action == 0:
            # NOOP
            pass
        elif action == 1:
            _y += self.step_size
        elif action == 2:
            _x += self.step_size
        elif action == 3:
            _y -= self.step_size
        elif action == 4:
            _x -= self.step_size
        _x = np.clip(_x, self.dot_size[0], self.window_width - 1 - self.dot_size[0])
        _y = np.clip(_y, self.dot_size[1], self.window_height - 1 - self.dot_size[1])
        self.pos = (_x, _y)


class MovingDotContinuousEnv(MovingDotEnv):
    """ 
    Continuous Action MovingDot env 
    
    Args:

        step_size: amount of pixels to move dot if component direction is greater than moving_thd
        low: negative number specifying how the dot can move up/left
        high: positive number specifying how the dot can move down/right
        moving_thd: absolute minimum value of each action component to move the dot along that direction
        -- Parent Class Arguments --
        render_mode: chose environemnt render mode - human or rgb_array
        random_start: if True, dot randomly starts on canvas, or in top left if False
        max_steps: maximum number of steps in an episode before truncation occurs
    """
    def __init__(self, low=-1, high=1, moving_thd=0.3, step_size=1, **kwargs):  # moving_thd is empirically determined
        super(MovingDotContinuousEnv, self).__init__(**kwargs)
        assert moving_thd >= 0.1, "moving_thd must be >= 0.01"

        self.step_size = step_size
        self._low, self._high = (low, high)

        self._moving_thd = moving_thd  # used to decide if the object has to move, see step func below.
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

    def _update_pos(self, action):
        a_x, a_y = action
        assert self._low <= a_x <= self._high, f"movement along x-axis has to fall in between {self._low} to {self._high}"
        assert self._low <= a_y <= self._high, f"movement along y-axis has to fall in between {self._low} to {self._high}"

        """
        [Note]
        Since the action values are continuous for each x/y pos,
        we round the position of the object after executing the action on the 2D space.
        """
        new_x = self.pos[0] + self.step_size if a_x >= self._moving_thd else self.pos[0] - self.step_size
        new_y = self.pos[1] + self.step_size if a_y >= self._moving_thd else self.pos[1] - self.step_size

        _x = np.clip(new_x,
                              self.dot_size[0], self.window_width - 1 - self.dot_size[0])
        _y = np.clip(new_y,
                              self.dot_size[1], self.window_height - 1 - self.dot_size[1])
        self.pos = (_x, _y)
