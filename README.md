# [Heeding the Inner Voice: Aligning ControlNet Training via Intermediate Features Feedback]()

<a href=""><img src="https://img.shields.io/badge/arXiv-2505.21144-b31b1b.svg" height=22.5><a>
<a href="inference.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>
<a href=""><img src="https://img.shields.io/badge/Project-Website-blue" height=22.5><a>
[![License](https://img.shields.io/github/license/AIRI-Institute/al_toolbox)](./LICENSE.txt)

<p align="center">
  <img src="images/results.png" 
       alt="Method Diagram" 
       width="1000" 
       title="System Architecture Overview">
</p>


Despite significant progress in text-to-image diffusion models, achieving precise spatial control over generated outputs remains challenging. One of the popular approaches for this task is ControlNet, which introduces an auxiliary conditioning module into the architecture. To improve alignment of the generated image and control, ControlNet++ proposes a cycle consistency loss to refine correspondence between controls and outputs, but restricts its application to the final denoising steps, while the main structure is introduced at an early generation stage. To address this issue, we suggest **InnerControl** -- a training strategy that enforces spatial consistency across all diffusion steps. Specifically, we train lightweight control prediction probes — small convolutional networks — to reconstruct input control signals (e.g., edges, depth) from intermediate UNet features at every denoising step. We prove the efficiency of such models to extract signals even from very noisy latents and utilize these models to generate pseudo ground truth controls during training. Suggested approach enables alignment loss that minimizes the difference between predicted and target condition throughout the whole diffusion process. Our experiments demonstrate that our method improves control alignment and fidelity of generation. By integrating this loss with established training techniques (e.g., ControlNet++), we achieve high performance across different condition methods such as edge and depth conditions.

## Environments
```bash
git clone https://github.com/control/InnerControl.git
pip3 install -r requirements.txt
pip3 install clean-fid
pip3 install torchmetrics
```

## 📌 Data Preperation
**All the organized data has been put on Huggingface and will be automatically downloaded during training or evaluation.** You can preview it in advance to check the data samples and disk space occupied with following links.
|   Task    | Training Data 🤗 | Evaluation Data 🤗 |
|:----------:|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
|  LineArt, Hed   | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_train), 1.14 TB | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_canny_eval), 2.25GB |
|  Depth   |  [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth), 1.22 TB | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth_eval), 2.17GB |

## Quickstart

### Jupyter notebook
We provide example of applying our pretrained model to generate images in the [notebook]().

### 📌 Method diagram 

<p align="center">
  <img src="images/pipeline.png" 
       alt="Method Diagram" 
       width="1000" 
       title="System Architecture Overview">
</p>


### 📌 Training
By default, we conduct our training on 8 A100-80G GPUs. You can change number of utilized gpu number in [train/config.yaml](train/config.yml) file. If you lack sufficient computational resources, you can reduce the batch size while increasing gradient accumulation.

We can directly perform reward-alignment fine-tuning.

```bash
bash train/aligned_depth.sh
bash train/aligned_hed.sh
bash train/aligned_linedrawing.sh
```

### 📌 Evaluation
### Checkpoints Preparation
Please download the model weights and put them into each subset of `checkpoints`:
|   model    | ControlNet weights  | Align model                                                                 |
|:----------:|:--------------:|:----------------------------------------------------------------------|
|  LineArt   | [model](https://drive.google.com/drive/folders/1_K9NlB4iqZOJMxKM8IYFY_AhNLZyYkYU?usp=sharing) | [model](https://drive.google.com/file/d/156qGQPnAXAgZTm-t9Z-JYPJUkeocQmfk/view?usp=sharing)
|  Depth   |  [model](https://drive.google.com/drive/folders/12DmxpUXQTOw7L73kYU6yPTp1QlblgCh4?usp=sharing) | [model](https://drive.google.com/file/d/1bweES5Sf23mcEtdwLH0dVzWujIhFXWVf/view?usp=sharing)
|  Hed (SoftEdge)   | [model](https://drive.google.com/drive/folders/1LAQ0iIT6YsIWg_rAf1gKXriKUPRrIajo?usp=sharing) |[model](https://drive.google.com/file/d/1HElEu7mDoNvqAkHlGBjhPLrCiPBJMye9/view?usp=sharing) 
|

### 📌 Evaluate Controllability
Please make sure the folder directory is consistent with the test script, then you can eval each model by:
```bash
bash eval/eval_depth.sh
bash eval/eval_hed.sh
bash eval/eval_linedrawing.sh
```

### 📌 Evaluate CLIP-Score and FID
To evaluate CLIP and FID:


```bash
bash eval/eval_clip.sh
bash eval/eval_fid.sh
```

For FID evaluation you should additionally save dataset images into separate folder.


## 🙏 Acknowledgements
We sincerely thank the [Huggingface](https://huggingface.co), [ControlNet](https://github.com/lllyasviel/ControlNet), [ControlNet++](https://github.com/liming-ai/ControlNet_Plus_Plus) and [Readout Guidance](https://github.com/google-research/readout_guidance) communities for their open source code and contributions. Our project would not be possible without these amazing works.

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```

```
