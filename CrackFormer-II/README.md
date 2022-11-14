

# CrackFormer

>[CrackFormer: Transformer Network for Fine-Grained Crack Detection
>Huajun Liu, Xiangyu Miao, Christoph Mertz, Chengzhong Xu, Hui Kong; 
ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_CrackFormer_Transformer_Network_for_Fine-Grained_Crack_Detection_ICCV_2021_paper.html)

## usage
### datasets
Download the [cracktree](https://www.sciencedirect.com/science/article/pii/S0167865511003795),[crackls315](Deepcrack: Learning hierarchi-
cal convolutional features for crack detection.),[stone331](https://www.sciencedirect.com/science/article/pii/S1051200420302529),[crack537](https://www.sciencedirect.com/science/article/pii/S0925231219300566) dataset and the file follows the following structure.

```
|-- datasets
    |-- crack315
        |-- train
        |   |-- train.txt
        |   |--img
        |   |   |--<crack1.jpg>
        |   |--gt
        |   |   |--<crack1.bmp>
        |-- valid
        |   |-- Valid_image
        |   |-- Lable_image
        |   |-- Valid_result
        ......
```

train.txt format
```
./dataset/crack315/img/crack1.jpg ./dataset/crack315/gt/crack1.bmp
./dataset/crack315/img/crack2.jpg ./dataset/crack315/gt/crack2.bmp
.....
```

### Valid

Change the 'pretrain_dir','datasetName' and 'netName' in test.py

```
python test.py
```

### Pre-trained model

Download the trained model.

```
|-- model
    |-- <netname>
        |   |-- <trained_model.pkl>
        ......
```



| dataset    | Pre-trained Model                                            |
| ---------- | ------------------------------------------------------------ |
| cracktree  | [Link](https://drive.google.com/file/d/1avhmDO7AdM_D4BR25aeNg7lBgch6m_d8/view?usp=share_link) |
| crackls315 | [Link](https://drive.google.com/file/d/1ugZkeQ8_RFkaP_caLCIshbzJDD2MOHau/view?usp=share_link) |
| stone331   | [Link](https://drive.google.com/file/d/1L_f-yUIQc1YP7xHaQtCn-agcVtjWVV6O/view?usp=share_link) |
| crack537   | [Link](https://drive.google.com/file/d/1A5m_rsFcwONii1fi_39jZ-HcTeLw7Rvb/view?usp=share_link) |



### Reproduced model

We reproduce two classical cracksegmentation models SDDNet and STRNet and the journal version of the paper has the latest experiments. The code and checkpoint can be obtained from the following. In order to get better results from the model on some of the dataset we used, we made some adjustments(Includes increasing the number of channels and replacing 16x downsampling with 8x downsampling) to the reproduced model.

[SDDNet: Real-Time Crack Segmentation](https://ieeexplore.ieee.org/abstract/document/8863123)



| dataset    | [SDDNet](https://drive.google.com/file/d/1Q72L4nR6kLpW2N9u0IlqpvAO522jc50z/view?usp=share_link)(Original) | [SDDNet](https://drive.google.com/file/d/1en8kssPcETVlwO7m92HSsk2dj9itRwuD/view?usp=share_link)(Modified) |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cracktree  | [Link](https://drive.google.com/file/d/11au0082GzjGG294HPE-JKcZsHRhZdLY1/view?usp=share_link) | [Link](https://drive.google.com/file/d/1nB9qVkJ7-JGWAEHkpMHvz58aZPgulDVz/view?usp=share_link) |
| crackls315 | [Link](https://drive.google.com/file/d/1-1OXGChYdyToNeuhn5cDNqQhCLnYKMYg/view?usp=share_link) | [Link](https://drive.google.com/file/d/1kU4-tJFx7S8d51Dkqt071gyLQaJlxWoP/view?usp=share_link) |
| stone331   | [Link](https://drive.google.com/file/d/1k1C36Mj0mg9rLItKPAlRocXO1fBY_bSz/view?usp=share_link) | [Link](https://drive.google.com/file/d/15OR5Bpp3h_lWbOa5If_pF6WKD0stqlr4/view?usp=share_link) |
| crack537   | [Link](https://drive.google.com/file/d/1a9O8HWhv5Yfv8ndqPLzzgljD6eQPY22y/view?usp=share_link) | /                                                            |


[STRNet: Efficient attention-based deep encoder and decoder for automatic crack segmentation](https://journals.sagepub.com/doi/full/10.1177/14759217211053776)

| dataset        | [STRNet](https://drive.google.com/file/d/15vtHFKvyWNOsWi3f1PJcjR-Nr9DMh2V4/view?usp=share_link)(Original) | [STRNet](https://drive.google.com/file/d/15QDEAWWwaI62sDB7oz_otSXQNTXH-Lhj/view?usp=share_link)(Modified) |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cracktree      | [Link](https://drive.google.com/file/d/1L-B_45RQqe616lUW3NxD8qO8ngVo2cpa/view?usp=share_link) | [Link](https://drive.google.com/file/d/1gL7Oy49ZjujIHAwqRl6KCaktl9-siCIN/view?usp=share_link) |
| crackls315     | [Link](https://drive.google.com/file/d/1mErEjzODJB8LXxNo0Z_FIG9n6xVaN9oW/view?usp=share_link) | [Link](https://drive.google.com/file/d/1g39lXV9h0_pdJ7P3ZkXMA2cdXwZ12W1a/view?usp=share_link) |
| stone331       | [Link](https://drive.google.com/file/d/1IXKq5vOPSIhg57LV_8ltzvsJ3Odie-t1/view?usp=share_link) | [Link](https://drive.google.com/file/d/1g39lXV9h0_pdJ7P3ZkXMA2cdXwZ12W1a/view?usp=share_link) |
| crack537       | [Link](https://drive.google.com/file/d/1iS-JnlpfQCm4IKsWGaa4KYbzrozMdZMZ/view?usp=share_link) | /                                                            |
| crackbenchmark | [Link](https://drive.google.com/file/d/1Au66qlYGIhmdNGTrhIo5ADNz4VgxiPPr/view?usp=share_link) | /                                                            |

## Acknowledgments

- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.

## Reference
>``` 
>@InProceedings{Liu_2021_ICCV,
>author    = {Liu, Huajun and Miao, Xiangyu and Mertz, Christoph and Xu, Chengzhong and Kong, Hui},
>title     = {CrackFormer: Transformer Network for Fine-Grained Crack Detection},
>booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
>month     = {October},
>year      = {2021},
>pages     = {3783-3792}
>}
>
>@article{choi2019sddnet,
>title={SDDNet: Real-time crack segmentation},
>author={Choi, Wooram and Cha, Young-Jin},
>journal={IEEE Transactions on Industrial Electronics},
>volume={67},
>number={9},
>pages={8016--8025},
>year={2019},
>publisher={IEEE}
>}
>
>@article{kang2022efficient,
>title={Efficient attention-based deep encoder and decoder for automatic crack segmentation},
>author={Kang, Dong H and Cha, Young-Jin},
>journal={Structural Health Monitoring},
>volume={21},
>number={5},
>pages={2190--2205},
>year={2022},
>publisher={SAGE Publications Sage UK: London, England}
>}
>
>```
