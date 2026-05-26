import json
import os
import threading
from pathlib import Path
from mikazuki.log import log


class Config:

    def __init__(self, path: str):
        self.path = path
        self._stored = {}
        self._default = {
            "last_path": "",
            "saved_params": {},
            "active_ui_profile": "builtin-legacy",
            "plugin_developer_mode": False,
        }
        self._lock = threading.Lock()

    def load_config(self):
        log.info(f"Loading config from {self.path}")
        if not os.path.exists(self.path):
            with self._lock:
                self._stored = dict(self._default)
            self.save_config()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    loaded = {}
                with self._lock:
                    self._stored = {**self._default, **loaded}
        except Exception as e:
            log.error(f"Error loading config: {e}")
            with self._lock:
                self._stored = dict(self._default)
            return

    def save_config(self):
        with self._lock:
            snapshot = dict(self._stored)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log.error(f"Error saving config: {e}")

    def __getitem__(self, key):
        with self._lock:
            return self._stored.get(key, None)

    def __setitem__(self, key, value):
        with self._lock:
            self._stored[key] = value


app_config = Config(Path(__file__).parents[2].absolute() / "assets" / "config.json")
