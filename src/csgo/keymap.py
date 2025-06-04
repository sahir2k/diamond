CSGO_KEYMAP = {
    "w": "up",
    "d": "right",
    "a": "left",
    "s": "down",
    "space": "jump",
    "left ctrl": "crouch",
    "left shift": "walk",
    "1": "weapon1",
    "2": "weapon2",
    "3": "weapon3",
    "r": "reload",

    # Override mouse movement with arrows
    "up": "camera_up",
    "right": "camera_right",
    "left": "camera_left",
    "down": "camera_down",
}


CSGO_FORBIDDEN_COMBINATIONS = [
    {"up", "down"},
    {"left", "right"},
    {"weapon1", "weapon2"},
    {"weapon1", "weapon3"},
    {"weapon2", "weapon3"},
    {"camera_up", "camera_down"},
    {"camera_left", "camera_right"},
]
