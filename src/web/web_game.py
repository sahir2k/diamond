import asyncio
import base64
import io
import json
from typing import Tuple, Union

from PIL import Image
from websockets.server import WebSocketServerProtocol, serve
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial

from csgo.action_processing import CSGOAction
from game.play_env import PlayEnv
from game.dataset_env import DatasetEnv


JS_TO_KEY = {
    "KeyW": "w",
    "KeyA": "a",
    "KeyS": "s",
    "KeyD": "d",
    "Space": "space",
    "ControlLeft": "left ctrl",
    "ShiftLeft": "left shift",
    "Digit1": "1",
    "Digit2": "2",
    "Digit3": "3",
    "KeyR": "r",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
}


class WebGame:
    def __init__(self, env: Union[PlayEnv, DatasetEnv], size: Tuple[int, int], mouse_multiplier: int, fps: int, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.env = env
        self.height, self.width = size
        self.mouse_multiplier = mouse_multiplier
        self.fps = fps
        self.host = host
        self.port = port

    def encode_obs(self, obs):
        assert obs.ndim == 4 and obs.size(0) == 1
        img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1,2,0).cpu().numpy())
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    async def handler(self, websocket: WebSocketServerProtocol):
        obs, _ = self.env.reset()
        keys_pressed = set()
        l_click = False
        r_click = False
        mouse_x = 0
        mouse_y = 0
        await websocket.send(self.encode_obs(obs))
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=1 / self.fps)
                    event = {} if msg is None else json.loads(msg)
                except asyncio.TimeoutError:
                    event = None
                if event:
                    etype = event.get("type")
                    if etype == "key_down":
                        key = JS_TO_KEY.get(event.get("code"))
                        if key:
                            keys_pressed.add(key)
                    elif etype == "key_up":
                        key = JS_TO_KEY.get(event.get("code"))
                        if key and key in keys_pressed:
                            keys_pressed.remove(key)
                    elif etype == "mouse_move":
                        mouse_x = event.get("dx", 0) * self.mouse_multiplier
                        mouse_y = event.get("dy", 0) * self.mouse_multiplier
                    elif etype == "mouse_down":
                        if event.get("button") == 0:
                            l_click = True
                        if event.get("button") == 2:
                            r_click = True
                    elif etype == "mouse_up":
                        if event.get("button") == 0:
                            l_click = False
                        if event.get("button") == 2:
                            r_click = False
                    elif etype == "reset":
                        obs, _ = self.env.reset()
                        keys_pressed.clear()
                        l_click = r_click = False
                        mouse_x = mouse_y = 0
                action = CSGOAction(list(keys_pressed), mouse_x, mouse_y, l_click, r_click)
                next_obs, _, end, trunc, _ = self.env.step(action)
                if end or trunc:
                    next_obs, _ = self.env.reset()
                    keys_pressed.clear()
                    l_click = r_click = False
                    mouse_x = mouse_y = 0
                obs = next_obs
                await websocket.send(self.encode_obs(obs))
        except Exception:
            pass

    def run(self) -> None:
        self.env.print_controls()
        asyncio.run(self._run())

    async def _run(self) -> None:
        web_path = Path(__file__).resolve().parent.parent.parent / "web"
        handler = partial(SimpleHTTPRequestHandler, directory=str(web_path))
        http_server = ThreadingHTTPServer((self.host, self.port + 1), handler)

        async with serve(self.handler, self.host, self.port, ping_interval=None):
            print(f"Web UI listening on ws://{self.host}:{self.port}")
            print(f"Open http://{self.host}:{self.port + 1}/ in your browser")
            loop = asyncio.get_running_loop()
            server_task = loop.run_in_executor(None, http_server.serve_forever)
            try:
                await asyncio.Future()
            finally:
                http_server.shutdown()
                await server_task
