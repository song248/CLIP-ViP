# CLIP-ViP Video Feature Extractor

## Conda Env
```
conda create -n CLIP-ViP python=3.9 -y
conda activate CLIP-ViP
pip install -r requirements.txt
```
## Execute
```
python main.py --config configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json
```

## Use Shell
```
chmod +x start.sh
./start.sh
```
<br></br>
If you encounter an <b>apex</b> specific error
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```