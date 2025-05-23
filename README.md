## CS224R Default Project

Author: Feiyang Zhu, Shutong Zhang, Siyuan Wu

### Instllation
Install and activate the conda environment using the following command:

```
conda env create -f environment.yml
conda activate cs224r
```

### Data Generation
Generate dataset for SFT and DPO using the following command:
```
python preprocess_sft_datasets.py
python preprocess_ultrafeedback.py
```
Generated datasets will be stored in `./processed_dataset` folder.

### Supervised Finetuning
To finetune the model on the smoltalk dataset, run:
```
python train.py --config config/sft_smol.yaml # You may need to manually modify path the config file
```
To finetune the model on the warmstart dataset, run:
```
python train.py --config config/sft_warm.yaml # You may need to manually modify path the config file
```
Trained models will be saved in `./SFT/models` folder.

### Direct Preference Optimization
To run DPO on the ultrafeedback dataset, run:
```
python train.py # You may need to manually modify path in DPO/config/dpo.yaml
```
Trained models will be saved in `./DPO/ckpts` folder.

### Evaluation
To evaluate the trianed model on the ultrafeedback dataset, run:
```
python eval_ultrafeedback.py --model_path $model_path --dataset_path $dataset_path
```
To evaluate the trianed model on the countdown dataset, run:
```
python eval_countdown.py --model_path $model_path --dataset_path $dataset_path
```

### Extention
- We implemented the first extention in `eval/inference_external_tool.py`, it is able to run countdown evaluation with the help of external tools.
- More to be added...
