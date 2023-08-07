# instruct-reid

This repo provides a basic training and testing framework for instruction guided person re-identification (instruct-ReID). 

### Installation

1, Download the transformers installation package from [download](https://github.com/huggingface/transformers)  
2, Add code 'image_features_n = self.visual_projection(vision_outputs[0])' after line 1077 and change the line 1079 code 'return image_features' to 'return image_features, image_features_n' in file './transformers/src/transformers/models/clip/modeling_clip.py'  
3, cd transformers & python setup.py install  
4, Download CLIP installation package from [download](https://github.com/openai/CLIP)  
5, cd CLIP & python setup.py install  
6, other requirements  

```
ftfy
regex
tqdm
torch
torchvision
socket
sklearn
opencv
```

### Prepare Pre-trained Models
```shell
mkdir logs && cd logs && mkdir pretrained
download pretrained model deit_base_distilled_patch16_224-df68dfff.pth and ViT-B-32.pt to pretrained directory
```
The file tree should be
```
logs
└── pretrained
    └── deit_base_distilled_patch16_224-df68dfff.pth
    └── ViT-B-32.pt
```
mkdir fashion_clip_model and download fashion_clip pretrained [model](https://github.com/patrickjohncyh/fashion-clip).

The file tree should be
```
fashion_clip_model
└── config.json
└── pytorch_model.bin
└── merges.txt
└── preprocessor_config.json
└── special_tokens_map.json
└── tokenizer_config.json
└── tokenizer.json
└── vocab.json
```

### Prepare data
shell
mkdir data
cp the dataset and annotation datalist to data directory.
We provide OmniReID annotation datalist download [link](https://drive.google.com/file/d/1d51ENyfMjdVwfLVmdkWSnndokg3Ym6wy/view?usp=drive_linkP)

The file tree should be
```
data
└── cuhk
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── cuhk03_1
└── ltcc
    └── croped_clothes
    └── datalist
        └── query_sc.txt
        └── gallery_sc.txt
        └── query_cc.txt
        └── gallery_cc.txt
        └── query_general.txt
        └── gallery_general.txt
        └── train.txt
    └── LTCC_ReID
    └── templates
    └── white_shirt.jpg
└── market
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── Market-1501
└── msmt
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── MSMT17_V1
└── prcc
    └── croped_clothes
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── rgb
    └── prcc_A_templates
    └── white_shirt.jpg
└── real1
    └── COCAS
    └── datalist
        └── runner_real1_v1_gpt.json
        └── train_attr.txt
        └── train_ctcc.txt
└── real2
    └── real_reid_image_face_blur
    └── datalist
        └── runner_real2_v1_gpt.json
        └── query_attr.txt
        └── gallery_attr.txt
        └── query.txt
        └── gallery.txt
└── vc_clothes
    └── croped_image
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── gallery
    └── query
    └── train
    └── white_shirt.jpg
```

### Training

```shell
./scripts/market/train.sh transformer_dualattn ${gpu_num} ${description}
```

### Testing

```shell
./scripts/test.sh transformer_dualattn ${/PATH/TO/YOUR/MODEL/} ${query-txt} ${gallery-txt} ${root_path} ${test_task_type}# default 1 GPUs
```

### inference model
We provide inference model for each task at [link](https://github.com/openai/CLIP).

### Demo
Trad-ReID  
![image](https://github.com/instruct-ReID/instructReID/blob/main/trad_reid.gif)  
CC-ReID  
![image](https://github.com/instruct-ReID/instructReID/blob/main/cc_reid.gif)  
CTCC-ReID  
![image](https://github.com/instruct-ReID/instructReID/blob/main/ctcc_reid.gif)  
LI-ReID  
![image](https://github.com/instruct-ReID/instructReID/blob/main/li_reid.gif)
