from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import asyncio
import base64
import torch # For dummy observations
import json # For parsing WebSocket messages

# Assuming game.py is structured to allow this import
from src.game.game import Game
# from src.game.play_env import PlayEnv # Too complex for now
# from src.csgo.action_processing import CSGOAction # Game handles this internally

app = FastAPI()

# Mock Environment for Game
class MockEnv:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.action_space_shape = (9,) # Dummy shape

    def reset(self):
        # obs shape (batch, channels, height, width), float, range [-1, 1]
        dummy_obs = torch.rand(1, 3, self.screen_height, self.screen_width) * 2 - 1
        return dummy_obs, {"header": [["MockEnv Initialized"]]}

    def step(self, action):
        dummy_obs = torch.rand(1, 3, self.screen_height, self.screen_width) * 2 - 1
        dummy_reward = torch.tensor([0.0])
        dummy_end = torch.tensor([False])
        dummy_trunc = torch.tensor([False])
        dummy_info = {"header": [["MockEnv Step"]]}
        return dummy_obs, dummy_reward, dummy_end, dummy_trunc, dummy_info

    def print_controls(self):
        print("MockEnv has no specific controls to print via this method.")

    def next_mode(self): return False
    def next_axis_1(self): return False
    def prev_axis_1(self): return False
    def next_axis_2(self): return False
    def prev_axis_2(self): return False


# Global Game instance
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
mock_env = MockEnv(SCREEN_WIDTH, SCREEN_HEIGHT)
# Game(play_env, size, mouse_multiplier, fps (removed), verbose)
game_instance = Game(
    play_env=mock_env,
    size=(SCREEN_HEIGHT, SCREEN_WIDTH), # Note: Game expects (height, width)
    mouse_multiplier=1,
    # fps=30, # fps parameter removed from Game
    verbose=True,
)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>FastAPI Web Server</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f0f0f0; display: flex; flex-direction: column; align-items: center; }
        h1 { color: #333; }
        #video_container { border: 1px solid #ccc; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        #video_stream_image { display: block; }
        #controls_info { margin-top:20px; padding:15px; background-color: #fff; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        #ws_status { margin-top: 10px; font-style: italic; }
    </style>
</head>
<body>
    <h1>CSGO AI Stream</h1>
    <div id="video_container">
        <img id="video_stream_image" src="/stream_placeholder.jpg" width="640" height="480" alt="Video Stream" />
    </div>
    <div id="controls_info">
        <h2>Controls:</h2>
        <p>Movement: W, A, S, D</p>
        <p>Jump: Space</p>
        <p>Pause/Unpause: P</p>
        <p>Reset Game: R</p>
        <p>Mouse clicks and precise aiming are not yet implemented in this interface.</p>
    </div>
    <div id="ws_status">WebSocket Status: Connecting...</div>

    <script>
        const videoStreamElement = document.getElementById('video_stream_image');
        const wsStatusElement = document.getElementById('ws_status');
        let socket = null;

        // Initialize WebSocket
        function setupWebSocket() {
            const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = wsProtocol + "//" + window.location.host + "/ws/input";
            socket = new WebSocket(wsUrl);

            socket.onopen = () => {
                console.log("WebSocket connection established.");
                wsStatusElement.textContent = "WebSocket Status: Connected";
                sendInput(); // Send initial state
            };

            socket.onclose = (event) => {
                console.log("WebSocket connection closed.", event);
                wsStatusElement.textContent = "WebSocket Status: Closed. Attempting to reconnect...";
                // Simple reconnect logic
                setTimeout(setupWebSocket, 3000);
            };

            socket.onerror = (error) => {
                console.error("WebSocket error:", error);
                wsStatusElement.textContent = "WebSocket Status: Error (see console)";
            };
        }

        // SSE Handling for video stream
        if (!!window.EventSource) {
            const eventSource = new EventSource("/stream");
            eventSource.onmessage = (event) => {
                // Assuming server sends full "data:image/jpeg;base64,..." string
                videoStreamElement.src = event.data;
            };
            eventSource.onerror = (error) => {
                console.error("EventSource failed:", error);
                videoStreamElement.alt = "Stream failed to load. Check console.";
                eventSource.close();
            };
        } else {
            videoStreamElement.alt = "Your browser doesn't support Server-Sent Events.";
        }

        // Keyboard Input Capture
        const pressedKeys = {};
        let pauseNextSend = false;
        let resetNextSend = false;

        const gameKeys = ['w', 'a', 's', 'd', ' ', 'p', 'r']; // Add other game keys if needed

        window.addEventListener('keydown', (event) => {
            const key = event.key.toLowerCase();
            if (gameKeys.includes(key)) {
                event.preventDefault(); // Prevent default browser action for game keys
            }

            if (key === 'p') {
                pauseNextSend = true;
            } else if (key === 'r') {
                resetNextSend = true;
            } else {
                pressedKeys[key] = true;
            }
            sendInput();
        });

        window.addEventListener('keyup', (event) => {
            const key = event.key.toLowerCase();
            if (key !== 'p' && key !== 'r') { // Toggle keys are momentary
                pressedKeys[key] = false;
            }
            sendInput();
        });

        // Mouse input (placeholders for now)
        // window.addEventListener('mousemove', (event) => { /* ... update mouse_x, mouse_y ... sendInput(); */ });
        // window.addEventListener('mousedown', (event) => { /* ... update l_click/r_click ... sendInput(); */ });
        // window.addEventListener('mouseup', (event) => { /* ... update l_click/r_click ... sendInput(); */ });


        function sendInput() {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const inputState = {
                    keys_pressed_map: { ...pressedKeys }, // Send a copy
                    mouse_x: 0, // Placeholder
                    mouse_y: 0, // Placeholder
                    l_click: false, // Placeholder
                    r_click: false, // Placeholder
                    pause_toggle: pauseNextSend,
                    reset_game: resetNextSend
                };
                socket.send(JSON.stringify(inputState));

                // Reset toggle flags after sending
                if (pauseNextSend) pauseNextSend = false;
                if (resetNextSend) resetNextSend = false;
            }
        }

        // Initial setup
        setupWebSocket();

    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return HTMLResponse(content=html_content)


async def frame_generator():
    global game_instance
    game_runner = game_instance.run() # Get the generator
    try:
        while True:
            # Input is now handled by the WebSocket endpoint, so no call to game_instance.update_input here.
            # The game_instance.run() will proceed based on state updated by the WebSocket.
            yielded_data = await asyncio.to_thread(next, game_runner) # Run blocking generator in thread

            frame_bytes = yielded_data.get("frame")
            # header_info = yielded_data.get("header") # Can be used later
            # paused_state = yielded_data.get("paused")
            # done_state = yielded_data.get("done")

            if frame_bytes:
                frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
                sse_data = f"data:image/jpeg;base64,{frame_base64}\n\n" # Standard for base64 img in SSE
                yield sse_data

            await asyncio.sleep(0.01) # Adjust sleep time as needed for frame rate
    except StopIteration:
        print("Game loop finished.")
    except Exception as e:
        print(f"Error in frame_generator: {e}")
    finally:
        # Consider how to handle game closure or if game_runner needs explicit closing
        print("Stream ended.")


@app.get("/stream")
async def stream_video():
    return StreamingResponse(frame_generator(), media_type="text/event-stream")

@app.websocket("/ws/input")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # print(f"Received raw data: {data}") # For debugging
            try:
                input_data = json.loads(data)

                # Prepare the dictionary for game_instance.update_input()
                # This structure should match what Game.update_input expects
                parsed_input_for_game = {
                    'keys_pressed_map': input_data.get('keys_pressed_map', {}),
                    'mouse_x': input_data.get('mouse_x', 0),
                    'mouse_y': input_data.get('mouse_y', 0),
                    'l_click': input_data.get('l_click', False),
                    'r_click': input_data.get('r_click', False),
                    'pause_toggle': input_data.get('pause_toggle', False),
                    'reset_game': input_data.get('reset_game', False)
                }
                game_instance.update_input(parsed_input_for_game)
                # print(f"Processed input: {parsed_input_for_game}") # For debugging

            except json.JSONDecodeError:
                print(f"Error decoding JSON: {data}")
            except Exception as e:
                print(f"Error processing message: {e}")

    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        print("WebSocket connection closed")


if __name__ == "__main__":
    # Ensure correct uvicorn invocation if modules are in src
    # uvicorn.run("src.web_server:app", host="0.0.0.0", port=8000, reload=True)
    # Or run directly if PYTHONPATH is set up:
    uvicorn.run(app, host="0.0.0.0", port=8000)
