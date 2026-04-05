# RadarRift

Windows companion for **League of Legends** minimap tracking (overlay + capture).  
**Not affiliated with Riot Games.** Use at your own risk and follow [Riot’s Terms](https://www.riotgames.com/en/terms-of-use).

---

## Download (built app)

Use the packaged build — no Python install needed.

1. Open the repo on GitHub → **Releases** (right side) → pick the latest release.  
2. Download the attached **`.zip`** (portable build), extract it, run **`RadarRift.exe`**.

Direct link pattern (replace `YOUR_USER` with your GitHub username or org):

`https://github.com/YOUR_USER/RadarRift/releases/latest`

---

## Who can see the code?

This project lives on **GitHub**. **Only people with access to the repository** (e.g. public repo, or private repo + invited collaborators) can browse or clone the source.  
If the repo is **private**, release downloads are also limited to people GitHub allows to access that repo — to share builds widely without opening the code, host the zip on another file host and link it from here or from a separate public “downloads-only” repo.

---

## Run from source *(contributors / devs)*

Python **3.10+**, Windows.

1. `pip install -r requirements.txt`
2. `python main.py`

Follow in-app / `startup_check` prompts for models and cache.

---

## Build the `.exe` yourself

1. `pip install -r requirements-build.txt`
2. `python -m PyInstaller RadarRift.spec`

Output folder: **`dist/RadarRift/`**.

---

## License

Add a `LICENSE` file when you publish; until then, all rights reserved unless you state otherwise.
