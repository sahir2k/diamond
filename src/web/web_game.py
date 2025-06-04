import base64
import io
import json
from pathlib import Path
from typing import Tuple, Union

import asyncio
from threading import Thread

import websockets
from PIL import Image
from flask import Flask, request, send_from_directory

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

        self.keys_pressed = set()
        self.l_click = False
        self.r_click = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.obs, _ = self.env.reset()

        web_path = Path(__file__).resolve().parent.parent.parent / "web"
        self.app = Flask(__name__, static_folder=str(web_path), static_url_path="")
        self._setup_routes()
        self.clients = set()

    def encode_obs(self, obs) -> bytes:
        assert obs.ndim == 4 and obs.size(0) == 1
        img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _setup_routes(self) -> None:
        app = self.app

        @app.route("/")
        def index():
            return send_from_directory(app.static_folder, "index.html")

        @app.route("/event", methods=["POST"])
        def event():
            data = request.get_json(force=True)
            etype = data.get("type")
            if etype == "key_down":
                key = JS_TO_KEY.get(data.get("code"))
                if key:
                    self.keys_pressed.add(key)
            elif etype == "key_up":
                key = JS_TO_KEY.get(data.get("code"))
                if key and key in self.keys_pressed:
                    self.keys_pressed.remove(key)
            elif etype == "mouse_move":
                self.mouse_x = data.get("dx", 0) * self.mouse_multiplier
                self.mouse_y = data.get("dy", 0) * self.mouse_multiplier
            elif etype == "mouse_down":
                if data.get("button") == 0:
                    self.l_click = True
                if data.get("button") == 2:
                    self.r_click = True
            elif etype == "mouse_up":
                if data.get("button") == 0:
                    self.l_click = False
                if data.get("button") == 2:
                    self.r_click = False
            elif etype == "reset":
                self.obs, _ = self.env.reset()
                self.keys_pressed.clear()
                self.l_click = self.r_click = False
                self.mouse_x = self.mouse_y = 0
            return "", 204

    async def _handle_ws(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                etype = data.get("type")
                if etype == "key_down":
                    key = JS_TO_KEY.get(data.get("code"))
                    if key:
                        self.keys_pressed.add(key)
                elif etype == "key_up":
                    key = JS_TO_KEY.get(data.get("code"))
                    if key and key in self.keys_pressed:
                        self.keys_pressed.remove(key)
                elif etype == "mouse_move":
                    self.mouse_x = data.get("dx", 0) * self.mouse_multiplier
                    self.mouse_y = data.get("dy", 0) * self.mouse_multiplier
                elif etype == "mouse_down":
                    if data.get("button") == 0:
                        self.l_click = True
                    if data.get("button") == 2:
                        self.r_click = True
                elif etype == "mouse_up":
                    if data.get("button") == 0:
                        self.l_click = False
                    if data.get("button") == 2:
                        self.r_click = False
                elif etype == "reset":
                    self.obs, _ = self.env.reset()
                    self.keys_pressed.clear()
                    self.l_click = self.r_click = False
                    self.mouse_x = self.mouse_y = 0
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)

    async def _frame_loop(self):
        delay = 1 / self.fps
        while True:
            action = CSGOAction(list(self.keys_pressed), self.mouse_x, self.mouse_y, self.l_click, self.r_click)
            next_obs, _, end, trunc, _ = self.env.step(action)
            if end or trunc:
                next_obs, _ = self.env.reset()
            self.obs = next_obs
            img = self.encode_obs(self.obs)
            self.mouse_x = 0
            self.mouse_y = 0
            for ws in list(self.clients):
                try:
                    await ws.send(img)
                except websockets.ConnectionClosed:
                    self.clients.discard(ws)
            await asyncio.sleep(delay)

    def run(self) -> None:
        self.env.print_controls()
        thread = Thread(target=self.app.run, kwargs={"host": self.host, "port": self.port + 1, "threaded": True}, daemon=True)
        thread.start()

        async def main():
            async with websockets.serve(self._handle_ws, self.host, self.port, max_size=None):
                await self._frame_loop()

        asyncio.run(main())

