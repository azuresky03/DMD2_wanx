# DMD2 for Wanx2.1

This repository implements the [DMD2](https://github.com/tianweiy/DMD2) technique for the Wanx2.1 video generation model. It provides an efficient way to distill Wanx2.1 to achieve faster inference with comparable generation quality.

## Implementation Improvements

Our implementation includes several improvements adapted specifically for video models (for lower memory consumption):
- Distillation based on cfg-distilled model
- LoRA for fake score function
- FSDP (Fully Sharded Data Parallel) with checkpointing
- Sequence parallelism

## Usage

### Training
```bash
run main/sh/debug.sh
```

### Inference
```bash
run inference.sh
```

## Example Results

- `results/In_a_la_step5_shift7_guide8.mp4`
- `results/Milk_an_step5_shift10_guide8.mp4`
- `results/Raindro_step5_shift7_guide8.mp4`
- `results/The_vet_step5_shift7_guide8.mp4`

## Known Limitations

### High Memory Consumption
Does not support high resolution (e.g., 720p 81 frames) training.

### Biased Distribution
Our experiments show that the learned distribution is biased and does not generalize well to resolutions different from training:
- `results/The_vet_step5_shift7_guide5.mp4` (selected lower quality case, 480p, same resolution as training)
- `results/The_vet_step5_shift7_guide5_720p.mp4` (average quality, 720p, different resolution)
- `results/Milk_an_step5_shift10_guide5_720p.mp4` (average quality, 720p, different resolution)

You can perceive an emergence of blinds as background in these cases.

### Unstable Convergence
When training with `main/sh/multi_node.sh` configurations, the results in the Example Results section are trained with 500 steps. When continued to 1000 steps, there is a tendency of overexposure. See `results/In_a_la_step5_shift7_guide8_1000steps.mp4`.

### Potential Solution: RL Post-Training
In combination with RL post-training, the distribution shift might be mitigated. See implementations in [distill_wan2.1](https://github.com/azuresky03/distill_wan2.1?tab=readme-ov-file#reinforcement-learning-rl) and results in the [Combined Techniques section](https://github.com/azuresky03/distill_wan2.1?tab=readme-ov-file#combined-techniques).