
<div align="center">

# MOE-FaceUM: A Unified Face Perception Model with Face-aware Mixture of Experts



> **<p align="justify"> Abstract:** *Recent advances in face perception tasks such as face recognition, emotion recognition, and age estimation have largely relied on task-specific models. Unified multi-task learning (MTL) frameworks based on Transformer backbones have emerged to leverage inter-task synergies through shared representations. However, different facial tasks often require task-specific features, which a single backbone struggles to fully capture. Inspired by the success of Mixture-of-Experts (MoE) architectures in large language models, we propose <i>MOE-FaceUM<i>, the first unified face analysis framework integrating MoE to simultaneously learn six facial tasks. Our model employs a \textbf{Face-aware Router} that dynamically directs inputs to task-relevant experts and emphasizes critical facial regions, enabling specialized feature extraction while maintaining efficiency. Comprehensive experiments on multiple benchmark datasets demonstrate the effectiveness of our approach. Notably, <i>MOE-FaceUM<i> improves binary facial attribute classification accuracy by approximately 4.4%, achieves lower normalized mean error (NME) in landmark localization, and maintains competitive performance in face parsing. For face recognition, especially on the challenging masked face dataset RMFD, our model enhancements outperforms the baseline by over 6%, showing enhanced robustness against intra-class variations and real-world noise. * </p>

# :rocket: News
- [07/31/2025] 🔥 We release <i>MOE-FaceUM</i> Inference Code.

## Installation
```bash
conda env create --file environment_facex.yml
conda activate facexformer

# Install requirements
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
## Download Models
The models can be downloaded manually from [Google Drive](https://huggingface.co/kartiknarayan/facexformer)

The directory structure should finally be:

```
  . ── MOE-FaceUM ──┌── ckpts/model.pt
                     ├── network
                     └── inference.py                    
```
## Usage

Download trained model from [oogle Drive](https://huggingface.co/kartiknarayan/facexformer) and ensure the directory structure is correct.<br>
For demo purposes, we have released the code for inference on a single image.<br>
It supports a variety of tasks which can be prompted by changing the "task" argument. 

```python
python inference.py --model_path ckpts/model.pt \
                    --image_path image.png \
                    --results_path results \
                    --task parsing \
                    --gpu_num 0


## TODOs
- Release dataloaders for the datasets used.
- Release training script.

# MOE-FaceUM
