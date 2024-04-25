# DAP
This is the code of the IJCAI 2024 paper: "**Dynamically Anchored Prompting for Task-Imbalanced Continual Learning.**"

**Abstract:** Existing continual learning literature relies heavily on a strong assumption that tasks arrive with a balanced data stream, which is often unrealistic in real-world applications. In this work, we explore task-imbalanced continual learning (TICL) scenarios where the distribution of task data is nonuniform across the whole learning process. We find that imbalanced tasks significantly challenge the capability of models to control the trade-off between stability and plasticity from the perspective of recent prompt-based continual learning methods. On top of the above finding, we propose Dynamically Anchored Prompting (DAP), a prompt-based method that only maintains a single general prompt to adapt to the shifts within a task stream dynamically. This general prompt is regularized in the prompt space with two specifically designed prompt anchors, called boosting anchor and stabilizing anchor, to balance stability and plasticity in TICL. Remarkably, DAP achieves this balance by only storing a prompt across the data stream, therefore offering a substantial advantage in rehearsal-free CL.


## Dependencies
    pytorch==1.12.1
    torchvision==0.13.1
    timm==0.6.7
    pillow==9.2.0
    matplotlib==3.5.3

## Usage
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        <ltcifar100_dap> \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --data-path /local_datasets/ \
        --output_dir ./output 

# Acknowledgement
We refer to the code architecture from [l2p-pytorch](https://github.com/JH-LEE-KR/l2p-pytorch). Many thanks to the authors.
