

# CrackFormer

>[CrackFormer: Transformer Network for Fine-Grained Crack Detection,
>Huajun Liu, Xiangyu Miao, Christoph Mertz, Chengzhong Xu, Hui Kong; 
ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_CrackFormer_Transformer_Network_for_Fine-Grained_Crack_Detection_ICCV_2021_paper.html)
>
>[CrackFormer Network for Pavement Crack Segmentation,
>Huajun Liu, Jing Yang, Xiangyu Miao, Christoph Mertz, Hui Kong; 
IEEE TITS 2023](https://ieeexplore.ieee.org/abstract/document/10109158)

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

                                                       |

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
>@article{Liu2023CrackFormer,
>title={CrackFormer Network for Pavement Crack Segmentation},
>author={Huajun Liu and Jing Yang and Xiangyu Miao and Christoph Mertz and Hui Kong},
>journal={IEEE Transactions on Intelligent Transportation Systems},
>volume={24},
>number={9},
>pages={9240-9252},
>year={2023},
>publisher={IEEE}
>}
>
>```
