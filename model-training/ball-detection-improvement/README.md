# Ball Detection Improvement (SoccerNet-only retraining)

This directory rebuilds the `ball-detection` project so we can **fine-tune the existing YOLO weights using only the SoccerNet Tracking dataset you provided** (unzipped under `F:/SoccerNetTasks/datasets`). There is no legacy/data-mix fallback anymore—every artifact in this folder assumes that single dataset as the source of truth.

The tracklet metadata provided for every SoccerNet sequence still looks like this:

```
trackletID_1 = player team left;10
trackletID_2 = player team left;30
...
trackletID_22 = goalkeepers team left;y
trackletID_25 = goalkeeper team right;X
trackletID_18 = ball;1
trackletID_14 = referee;main
```

We drop the jersey/role suffixes and collapse everything into the three classes required by the existing YOLO detection head:

- `Player` → everything containing `player` or `goalkeeper`
- `Ball`   → entries containing `ball`
- `Referee` → entries containing `referee`

> Need goalkeepers as a dedicated class? Extend `extraction.classes.order` to `["Player", "Goalkeeper", "Ball", "Referee"]` inside `yolo/configs/extraction.yaml` and update the keyword map—everything else keeps working.

## 📁 Folder layout

```
ball-detection-improvement/
├── README.md               # You are here
├── requirements.txt        # PyYAML, Pillow, tqdm for extraction helpers
└── yolo/
    ├── configs/            # Active configs loaded by the standalone pipeline
    ├── modules/            # Device, extractor, and trainer building blocks
    ├── utils/              # Config loader & logging helpers
    └── main.py             # Entry point for extraction + training
```

## 🟠 What is `ballDataset`?

`ballDataset/` is the YOLO-ready dataset that the pipeline writes directly under `model-training/ball-detection-improvement/`. It mirrors the standard structure:

- `ballDataset/images/train|val` – frames copied from SoccerNet sequences
- `ballDataset/labels/train|val` – YOLO txt files (class index + normalized boxes)
- `ballDataset/data.yaml` – the file passed to Ultralytics/YOLO during training

Every sample inside `ballDataset` traces back to the SoccerNet Tracking dump mounted at `F:/SoccerNetTasks/datasets` (the unzipped dataset you shared). No other sources or mixing steps are involved.

## ⚙️ Dataset extraction + training workflow

1. **Install the lightweight helpers** (needed for both extraction paths):

   ```powershell
   pip install -r model-training/ball-detection-improvement/requirements.txt
   ```

2. **Keep the configs pointing at your dataset**. By default `yolo/configs/paths.yaml` and `yolo/configs/extraction.yaml` already reference:
   - `workspace_root`: `c:/Users/kaan.aggunlu/Desktop/track/FoMAC/model-training/ball-detection-improvement`
   - `soccernet_root`: `f:/SoccerNetTasks/datasets`
   - `output_root` / `output.dataset_dir`: `ballDataset`

   Update those values only if you relocate the repo or the dataset drive letter.

3. **Run the standalone pipeline** (it handles extraction + training without any extra configs):

   ```powershell
   cd model-training/ball-detection-improvement/yolo
   python main.py --config-dir ./configs --run-extraction
   ```

   - Omit `--run-extraction` (or pass `--train-only`) if `ballDataset/` already exists.
   - Use `--extract-only` for a quick dataset refresh without kicking off training.

4. **Outputs** land under `model-training/ball-detection-improvement/models` and `.../reports` as configured in `yolo/configs/paths.yaml`. The trained weights live in `<models_root>/<project_name>/weights/*.pt` and the HTML reports/statistics mirror the structure under `reports_root`.

## 🔑 Key config knobs (all inside `yolo/configs`)

| File / key | Purpose | Default |
|------------|---------|---------|
| `paths.workspace_root` | Base folder for every relative path | Repo path shown above |
| `paths.soccernet_root` | Location of the extracted SoccerNet Tracking dataset | `f:/SoccerNetTasks/datasets` |
| `paths.dataset_yaml` | The `ballDataset/data.yaml` file fed into YOLO | `ballDataset/data.yaml` |
| `extraction.soccernet.splits.train` | Glob of sequences used for training | `train/SNMOT-*` |
| `extraction.soccernet.splits.val` | Glob of sequences used for validation | `test/SNMOT-*` |
| `extraction.output.dataset_dir` | Destination for frames/labels | `ballDataset` |
| `extraction.classes.keyword_map` | Maps SoccerNet descriptors to YOLO classes | `{player→Player, goalkeeper→Player, ball→Ball, referee→Referee}` |
| `yolo_params.model` | Warm-start weights (player detector) | `.../player-detection/.../best.pt` |
| `yolo_params.project_name` | Folder name under `models/` & `reports/` | `player_ball_detector_improvement` |

Tweak the globs if you need a custom split (e.g., reserved evaluation set), or bump hyper-parameters inside `yolo_params.yaml` for longer runs. Because the entire configuration now lives under `yolo/configs`, there is no chance of the old data-mix files sneaking back in.

## ✅ Why this setup

- **Single provenance:** everything inside `ballDataset` is derived solely from the SoccerNet Tracking dataset you attached—no hidden blending with legacy footage.
- **Drop-in compatibility:** the trainer still consumes a classic Ultralytics-style `data.yaml`, so you can re-use evaluation scripts, inference helpers, and checkpoints without edits.
- **Transparent relabeling:** by collapsing verbose tracklet descriptors via keyword matching, the model focuses on the Player/Ball/Referee classes we truly care about.

Happy retraining! ⚽️
