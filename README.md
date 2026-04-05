# RadarRift

Windows companion for **League of Legends** minimap tracking (overlay + capture).  
**Not affiliated with Riot Games.** Use at your own risk and follow [Riot’s Terms](https://www.riotgames.com/en/terms-of-use).

---

## Download (built app)

Use the packaged build — no Python install needed.

1. Open the repo on GitHub → **Releases** (right side) → pick the latest release.  
2. Download the attached **`.zip`** (portable build), extract it, run **`RadarRift.exe`**.

Latest release: **https://github.com/kagxyw/RadarRift/releases/latest**

---

## Run from source *(contributors / devs)*

Python **3.10+**, Windows.

1. `pip install -r requirements.txt`
2. `python main.py`

---

## Build the `.exe` yourself

1. `pip install -r requirements-build.txt`
2. `python -m PyInstaller RadarRift.spec`

Output folder: **`dist/RadarRift/`**.

---

## License

Add a `LICENSE` file when you publish; until then, all rights reserved unless you state otherwise.
