# CLIP-ViP Video Feature Extractor

```
conda create -n CLIP-ViP python=3.9 -y
conda activate CLIP-ViP
pip install -r requirements.txt
```
<br></br>
If you encounter an apex-specific error
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```