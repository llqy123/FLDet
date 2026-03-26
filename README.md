<h2 align="center">
✨Under Review
</h2>
Installation
Our project is developed on Detectron2.
Please follow the official installation instructions.

Data Preparation
Download the COCO dataset, and put it in the `datasets/` directory.
Download VL-PLM pre-generated pseudo-labeled data and our generated candidate pseudo-label data, and put them in the `datasets/open_voc` directory.
Dataset are organized in the following way:
```bazaar
datasets/
    coco/
        annotations/
            instances_train2017.json
            instances_val2017.json
            open_voc/
                instances_eval.json
                instances_train.json
        images/
            train2017/
                000000000009.jpg
                000000000025.jpg
                ...
            val2017/
                000000000776.jpg
                000000000139.jpg
                ...
        
```
Pseudo label generation
MarvelOVD dynamically learns open-vocabulary knowledge from offline-generated pseudo-labels under the guidance from the online training detector.
If necessary, please refer to pseudo label generation instruction to generate offline pseudo-labels.
Results on OVD-COCO
Mask R-CNN:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Novel AP</th>
<th valign="bottom">Base AP</th>
<th valign="bottom">Overall AP</th>
<!-- TABLE BODY -->
<!-- ROW: with LSJ -->
 <tr>
<td align="center">38.9</td>
<td align="center">56.4</td>
<td align="center">51.8</td>
</tr>
</tbody></table>
Training
We train the model under regular data augmentations (no Large Scale Jittering), without extra GPU memory occupation. (Runing on 4 GPUs with 24G Memory per GPU)
Training command
```bash 
python train_net.py --config configs/coco_ssod.yaml --num-gpus=4
```

Notes
The code is highly borrowed from VL_PLM, big thanks for the open-source commuity.
Questions and Issues, please contract wangk229@mail2.sysu.edu.cn
