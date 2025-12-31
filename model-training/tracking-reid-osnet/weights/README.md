# OSNet weights

The tracker reads OSNet weights from:

- `model-training/tracking/weights/osnet_x1_0.ts`

## Recommended (works without torchreid): TorchScript

Provide a TorchScript-exported OSNet model (`.ts`) at the path above.

In `model-training/tracking/config.yaml`:

- `osnet.enabled: true`
- `osnet.weights: "weights/osnet_x1_0.ts"` (relative to `config.yaml`)

## Alternative (requires deep-person-reid/torchreid): `.pth`

The official deep-person-reid model zoo provides OSNet checkpoints (Google Drive links).

- ImageNet-pretrained `osnet_x1_0`:
  - https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view

If you go this route, set in config:

- `osnet.weights: ".../osnet_x1_0.pth"`

…and ensure a compatible `torchreid` (deep-person-reid) install is available in the environment.
