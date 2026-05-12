# Music Energy & Stress Analysis

Visualises the **energy arc** of songs over time — showing how a track builds from its intro, peaks at the chorus, and winds down at the outro.

Given a YouTube playlist of public-domain music, the notebook downloads the audio, extracts acoustic features with [librosa](https://librosa.org/), and produces a three-panel chart per song plus a batch overlay for the full playlist.

---

## What it produces

**Per-song chart (3 panels)**

| Panel | What you see |
|---|---|
| Top | Stress score — raw (per frame ≈ 23 ms) and smoothed (5 s window) |
| Middle | Feature breakdown: RMS energy, spectral centroid, onset strength |
| Bottom | Mel spectrogram (full frequency picture over time) |

**Batch overlay** — all songs' smoothed stress curves on one plot, useful for comparing energy arcs across tracks.

---

## Stress score

The stress score is a weighted blend of three normalized features:

| Feature | Weight | What it captures |
|---|---|---|
| RMS energy | 50 % | Loudness / power |
| Spectral centroid | 25 % | Brightness (more treble = higher) |
| Onset strength | 25 % | Rhythmic density (attacks per second) |

---

## Setup

```bash
git clone https://github.com/fborbon/music-analytics.git
cd music-analytics
pip install yt-dlp librosa soundfile
```

Jupyter and the scientific stack (`numpy`, `scipy`, `matplotlib`) are assumed to be installed. If not:

```bash
pip install jupyterlab numpy scipy matplotlib
```

---

## Usage

Open the notebook and run cells top to bottom:

```bash
jupyter lab music_stress_analysis.ipynb
```

1. **Install** — installs missing dependencies into the kernel
2. **Config** — set `PLAYLIST_URL`, `N_SONGS`, and `SMOOTH_SEC`
3. **Download** — fetches audio as MP3 into `downloads/<playlist-id>/`; re-running skips already-downloaded tracks
4. **Analysis** — set `FILE_IDX` to pick a song, then run feature extraction and plot cells
5. **Batch** — overlays all songs' stress curves in one chart

> The notebook uses a `.yt-dlp-archive` file and `.features.npz` caches so re-runs skip completed steps automatically.

---

## Default playlist

The notebook is pre-configured with the [YouTube Audio Library — Free Public Domain Music](https://www.youtube.com/playlist?list=PLh5X0e7-mnI3lh-nb3fwZ8OcndGLfVdZX) playlist (334 tracks, CC0 / public domain). Swap `PLAYLIST_URL` for any other playlist.

---

## Tech stack

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — audio download
- [librosa](https://librosa.org/) — audio feature extraction
- [matplotlib](https://matplotlib.org/) — visualisation
- [scipy](https://scipy.org/) — signal smoothing
- [numpy](https://numpy.org/) — numerical core
