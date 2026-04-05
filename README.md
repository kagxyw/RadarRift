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

## Who can see the code?

**GitHub** controls this: **public** repo → anyone can browse source; **private** repo → only people you invite.  
On a **private** repo, **Releases** are also restricted to those users. To share a build publicly without opening the code, use a separate public “downloads” repo or another file host and link it here.

---

## Run from source *(contributors / devs)*

Python **3.10+**, Windows.

1. `pip install -r requirements.txt`
2. `python main.py`

---

## Build the `.exe` yourself

1. `pip install -r requirements-build.txt`
2. From the repo root: `python -m PyInstaller packaging/RadarRift.spec`

Output folder: **`dist/RadarRift/`**. Spec and ONNX runtime hook live under **`packaging/`**; bundled images and SVGs live under **`assets/`**.

### Maintenance CLIs *(from repo root)*

| Command | Purpose |
|--------|---------|
| `python -m tools.rebuild_cache` | Download skins/icons and rebuild identification matrices into `cache/` |
| `python -m tools.remake_onnx` | Export `.onnx` from Ultralytics `.pt` weights in `cache/` |
| `python -m tools.annotation_tool` | Tk minimap labelling UI (optional `path/to/images/train`) |
