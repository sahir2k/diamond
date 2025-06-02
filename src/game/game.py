from typing import Tuple, Union
import io # Added io.BytesIO

import numpy as np
# import pygame # Pygame removed
from PIL import Image

from csgo.action_processing import CSGOAction
from .dataset_env import DatasetEnv
from .play_env import PlayEnv


class Game:
    def __init__(
        self,
        play_env: Union[PlayEnv, DatasetEnv],
        size: Tuple[int, int], # Renamed to screen_height, screen_width for clarity
        mouse_multiplier: int,
        fps: int, # fps will be removed as it's Pygame dependent for clock.tick
        verbose: bool,
    ) -> None:
        self.env = play_env
        self.screen_height, self.screen_width = size # Store as screen_width, screen_height
        self.mouse_multiplier = mouse_multiplier
        # self.fps = fps # Removed, frame pacing handled by web server/client
        self.verbose = verbose

        self.env.print_controls() # This might print Pygame specific controls, review PlayEnv/DatasetEnv
        # print("\nControls:\n") # Pygame specific controls removed
        # print(" m  : switch control (human/replay)")
        # print(" .  : pause/unpause")
        # print(" e  : step-by-step (when paused)")
        # print(" ⏎  : reset env")
        # print("Esc : quit")
        # print("\n")
        # input("Press enter to start") # Removed blocking input

        # Input state variables to be updated by update_input
        self.keys_pressed_map = {}
        self.mouse_x = 0
        self.mouse_y = 0
        self.l_click = False
        self.r_click = False
        self.paused = False # Internal pause state
        self.should_reset = False # Signal for reset

    def update_input(self, input_data: dict) -> None:
        """
        Updates the game's input state based on data received from the web server.
        input_data is expected to be a dictionary with keys like:
        'keys_pressed_map': {'w': True, 'a': False, ...} (using characters for keys)
        'mouse_x': int (delta or absolute, TBD based on client)
        'mouse_y': int (delta or absolute, TBD based on client)
        'l_click': bool
        'r_click': bool
        'pause_toggle': bool (to toggle self.paused)
        'reset_game': bool (to set self.should_reset)
        """
        self.keys_pressed_map = input_data.get('keys_pressed_map', self.keys_pressed_map)
        # Assuming mouse inputs are deltas for now, like pygame.event.rel
        # If they are absolute, CSGOAction will need to handle that
        self.mouse_x = input_data.get('mouse_x', 0) * self.mouse_multiplier
        self.mouse_y = input_data.get('mouse_y', 0) * self.mouse_multiplier
        self.l_click = input_data.get('l_click', False)
        self.r_click = input_data.get('r_click', False)

        if input_data.get('pause_toggle', False):
            self.paused = not self.paused

        if input_data.get('reset_game', False):
            self.should_reset = True


    def run(self) -> None: # This will become a generator
        # pygame.init() # Pygame removed

        # header_height = 150 if self.verbose else 0 # verbose related drawing removed
        # header_width = 540
        # font_size = 16
        # screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) # Pygame removed
        # pygame.mouse.set_visible(False) # Pygame removed
        # pygame.event.set_grab(True) # Pygame removed
        # clock = pygame.time.Clock() # Pygame removed
        # font = pygame.font.SysFont("mono", font_size) # Pygame removed
        # x_center, y_center = screen.get_rect().center # Pygame removed
        # x_header = x_center - header_width // 2 # Pygame removed
        # y_header = y_center - self.height // 2 - header_height - 10 # Pygame removed
        # header_rect = pygame.Rect(x_header, y_header, header_width, header_height) # Pygame removed

        # def clear_header(): # Pygame removed
            # pygame.draw.rect(screen, pygame.Color("black"), header_rect)
            # pygame.draw.rect(screen, pygame.Color("white"), header_rect, 1)

        # def draw_text(text, idx_line, idx_column, num_cols): # Pygame removed
            # x_pos = 5 + idx_column * int(header_width // num_cols)
            # y_pos = 5 + idx_line * font_size
            # assert (0 <= x_pos <= header_width) and (0 <= y_pos <= header_height)
            # screen.blit(font.render(text, True, pygame.Color("white")), (x_header + x_pos, y_header + y_pos))

        # def draw_obs(obs, obs_low_res=None): # This logic will be moved into the class or a helper
            # assert obs.ndim == 4 and obs.size(0) == 1
            # img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            # pygame_image = np.array(img.resize((self.width, self.height), resample=Image.BICUBIC)).transpose((1, 0, 2))
            # surface = pygame.surfarray.make_surface(pygame_image)
            # screen.blit(surface, (x_center - self.width // 2, y_center - self.height // 2))

            # if obs_low_res is not None:
                # assert obs_low_res.ndim == 4 and obs_low_res.size(0) == 1
                # img = Image.fromarray(obs_low_res[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
                # h = self.height * obs_low_res.size(2) // obs.size(2)
                # w = self.width * obs_low_res.size(3) // obs.size(3)
                # pygame_image = np.array(img.resize((w, h), resample=Image.BICUBIC)).transpose((1, 0, 2))
                # surface = pygame.surfarray.make_surface(pygame_image)
                # screen.blit(surface, (x_header + header_width - w - 5, y_header + 5 + font_size))
                # screen.blit(surface, (x_center - w // 2, y_center + self.height // 2))

        # Internal state for the game loop
        current_obs, info = self.env.reset()
        ep_return = 0.0
        ep_length = 0

        # The main loop is now driven by external calls that also provide input.
        # It yields frames and status.
        # should_stop will be controlled by the web server (e.g., by stopping to call this generator)
        running = True
        while running: # This will be controlled by the web server stopping iteration
            if self.should_reset:
                current_obs, info = self.env.reset()
                ep_return = 0.0
                ep_length = 0
                self.should_reset = False
                # Reset internal input states related to a single action event
                self.l_click = False
                self.r_click = False
                # self.keys_pressed_map might persist depending on client updates

            if self.paused:
                # Yield current state even if paused, frame might be last frame or None
                frame_bytes, low_res_frame_bytes = self._process_obs_to_jpeg(current_obs, info.get("obs_low_res"))
                header_info = self._get_header_info(ep_return, ep_length, info)
                yield {"frame": frame_bytes, "low_res_frame": low_res_frame_bytes, "header": header_info, "paused": True}
                continue # Wait for next call to run (which might include new input to unpause)

            # Use self.keys_pressed_map, self.mouse_x, self.mouse_y, self.l_click, self.r_click
            # which are updated by update_input()
            # Note: CSGOAction will need to be adapted if keys_pressed_map uses chars like 'w'
            # instead of pygame.K_w constants. For now, we pass the map.
            csgo_action = CSGOAction(
                self.keys_pressed_map, self.mouse_x, self.mouse_y, self.l_click, self.r_click
            )

            # Reset mouse deltas after processing them for an action
            self.mouse_x = 0
            self.mouse_y = 0
            # Click states are also typically for a single frame/action
            # self.l_click = False # This depends on how client sends click, if it's held or per-event
            # self.r_click = False

            next_obs, rew, end, trunc, info = self.env.step(csgo_action)

            ep_return += rew.item()
            ep_length += 1

            current_obs = next_obs

            frame_bytes, low_res_frame_bytes = self._process_obs_to_jpeg(current_obs, info.get("obs_low_res"))
            header_info = self._get_header_info(ep_return, ep_length, info)

            yield {
                "frame": frame_bytes,
                "low_res_frame": low_res_frame_bytes,
                "header": header_info,
                "paused": False,
                "done": end or trunc
            }

            if end or trunc:
                current_obs, info = self.env.reset() # Reset for the next episode automatically
                ep_return = 0.0
                ep_length = 0
                # Potentially yield a "resetting" state or just start new episode in next iteration

        # pygame.quit() # Pygame removed
        self.env.close() # Ensure environment is closed if loop terminates

    def _process_obs_to_jpeg(self, obs, obs_low_res=None):
        """
        Converts observation tensor(s) to JPEG bytestring.
        """
        frame_bytes = None
        if obs is not None:
            # Current processing: obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            # This seems to be specific to a PyTorch tensor, ensure obs is in this format or adapt.
            # Assuming obs is already a NumPy array [H, W, C] or can be converted.
            # If obs is from self.env.reset() or self.env.step(), it might be a tensor.
            # For now, let's assume it's a NumPy array after the tensor processing.
            processed_obs_np = obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray(processed_obs_np)
            img_resized = img.resize((self.screen_width, self.screen_height), resample=Image.BICUBIC)
            byte_io = io.BytesIO()
            img_resized.save(byte_io, format='JPEG')
            frame_bytes = byte_io.getvalue()

        low_res_frame_bytes = None
        if obs_low_res is not None and self.verbose : # Only process if verbose
            processed_low_res_np = obs_low_res[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            img_low_res = Image.fromarray(processed_low_res_np)
            # Calculate proportional size for low_res
            h_orig, w_orig = processed_obs_np.shape[:2]
            h_low_orig, w_low_orig = processed_low_res_np.shape[:2]

            # Using self.screen_height and self.screen_width as the target size for the main observation
            h_new_low = self.screen_height * h_low_orig // h_orig
            w_new_low = self.screen_width * w_low_orig // w_orig

            img_low_res_resized = img_low_res.resize((w_new_low, h_new_low), resample=Image.BICUBIC)
            byte_io_low_res = io.BytesIO()
            img_low_res_resized.save(byte_io_low_res, format='JPEG')
            low_res_frame_bytes = byte_io_low_res.getvalue()

        return frame_bytes, low_res_frame_bytes

    def _get_header_info(self, ep_return, ep_length, env_info):
        """
        Constructs header information dictionary.
        """
        header_data = {
            "episode_return": f"{ep_return:.2f}",
            "episode_length": str(ep_length),
        }
        if self.verbose and env_info and "header" in env_info:
            # The original header from env_info is a list of lists of strings.
            # We can pass it as is, or reformat it.
            header_data["env_header"] = env_info["header"]
        return header_data

    # def reset(self): # This was a nested function, making it part of the class or handle directly in run
        # nonlocal obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click
        # obs, info = self.env.reset()
        # pygame.event.clear() # Pygame removed
        # do_reset = False
        # ep_return = 0
        # ep_length = 0
        # keys_pressed = [] # Will use self.keys_pressed_map
        # l_click = r_click = False

# if __name__ == "__main__": # Commented out as Game will be driven by web server
#     # This old way of running is not compatible with web server integration
#     # game = Game("human", tổng_reward=0) # Example parameters
#     # game.run()
#     pass
