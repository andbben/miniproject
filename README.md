# Disaster GAN - Satellite Image Generation and Damage Assessment

Generative AI pipeline for xBD disaster imagery with three stages:

1. Phase 1 trains a GAN to generate realistic pre-disaster satellite images.
2. Phase 2 trains a conditional GAN to translate pre-disaster scenes into post-disaster scenes.
3. Phase 3 trains a Siamese U-Net Transformer to localize buildings and classify damage.

## Repository Layout

Disaster Model GAN/
|-- configs/
|   `-- config.yaml
|-- checkpoints/
|   |-- pre_gan/
|   |-- post_gan/
|   `-- damage/
|-- data/
|   `-- xbd/
|-- outputs/
|-- src/
|   |-- data/
|   |   `-- datasets.py
|   |-- models/
|   |   |-- damage_assessment.py
|   |   `-- gan.py
|   |-- utils/
|   |   `-- metrics.py
|   |-- generate.py
|   |-- generate_and_assess.py
|   `-- train.py
|-- Siamese U-Net Transformer/
|   |-- model_siamese.py
|   |-- assess_siamese.py
|   |-- train_siamese.py
|   `-- test_siamese.py
|-- requirements.txt
`-- README(1).md

## Installation

Install dependencies:

pip install -r requirements.txt

## Dataset

The project expects the xBD dataset at:

data/xbd/
  train/
    images/
    labels/
  test/
    images/
    labels/

Update `data.xbd_root` in `configs/config.yaml` if needed.

## Training Commands

### Train all phases

python src/train.py --config configs/config.yaml --all

### Train phase 1 only Pre Disaster GAN

python src/train.py --config configs/config.yaml --phase 1

### Train phase 2 only Post-Disaster GAN

python src/train.py --config configs/config.yaml --phase 2

### Train phase 3 unified command

python src/train.py --config configs/config.yaml --phase 3

For Siamese specific interface with explicit override flags use:

python "Siamese U-Net Transformer/train_siamese.py" --config configs/config.yaml --epochs 60 --batch-size 8 --lr 1e-4


### Export xBD test pairs without assessment

python src/generate_and_assess.py --config configs/config.yaml --xbd_test --num_samples 4

## Assessment Commands

### Assess a real pre/post pair

python "Siamese U-Net Transformer/assess_siamese.py" --config configs/config.yaml --checkpoint checkpoints/damage/best.pt --pre_image data/noaa/pre.png --post_image data/noaa/post.png --tta


### Assess xBD test samples

python "Siamese U-Net Transformer/assess_siamese.py" --config configs/config.yaml --checkpoint checkpoints/damage/best.pt --xbd_test --num_samples 8 --output outputs/siamese_assessment --tta

### Generate GAN pairs and assess them directly

python "Siamese U-Net Transformer/assess_siamese.py" --config configs/config.yaml --checkpoint checkpoints/damage/best.pt --gan_generated --disaster hurricane --num_samples 4 --output outputs/siamese_gan_assessment --tta


## Output Artifacts

### GAN generation

- pre-disaster image PNG
- post-disaster image PNG
- side-by-side pair preview PNG

### Siamese assessment

- `<sample>_assessment.png`
- `<sample>_report.txt`

The assessment report includes:

- SDI
- total and affected area
- class-wise area breakdown
- mean IoU
- combined F1
- NDI change
- change magnitude

## Performance Notes

Hardware tuned @:

- GPU: RTX 2070 Super
- CPU: i7-9700K @ 4.6 GHz
- RAM: 32 GB DDR4 3000 MHz

Enabled optimizations include:

- AMP mixed precision on CUDA
- cuDNN benchmark mode
- fused AdamW on CUDA
- `pin_memory=True`
- persistent dataloader workers
- non-blocking host-to-device transfers
- channels-last memory format for CUDA models

## Typical Workflow

1. Train GANs with `src/train.py`.
2. Train the damage model with `src/train.py --phase 3`.
3. Generate pairs with `src/generate_and_assess.py`.
4. Assess damage with `Siamese U-Net Transformer/assess_siamese.py`.
5. If you want the transformer to score GAN outputs directly, use `--gan_generated`.
