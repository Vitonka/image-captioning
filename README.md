# Image Captioning
My PhD studies on the image captioning topic.

## Publications
[Text Augmentation Using BERT for Image Captioning](https://www.mdpi.com/2076-3417/10/17/5978/pdf)

```
@article{atliha2020text,
  title={Text Augmentation Using BERT for Image Captioning},
  author={Atliha, Viktar and {\v{S}}e{\v{s}}ok, Dmitrij},
  journal={Applied Sciences},
  volume={10},
  number={17},
  pages={5978},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

[Comparison of VGG and ResNet used as Encoders for Image Captioning](https://yadi.sk/d/szPwZTUt9XVqnw)

```
@inproceedings{atliha2020comparison,
  title={Comparison of VGG and ResNet used as Encoders for Image Captioning},
  author={Atliha, Viktar and {\v{S}}e{\v{s}}ok, Dmitrij},
  booktitle={2020 IEEE Open Conference of Electrical, Electronic and Information Sciences (eStream)},
  pages={1--4},
  year={2020},
  organization={IEEE}
}
```

## Instructions
To download and preprocess data from scratch do the following:
1. Run script `datasets/coco/download_raw.sh` to download raw dataset data
2. Create your own images preprocessing config and put it in the directory `datasets/coco/images/your_preprocessing_name/config.json` or use the existing ones at `datasets/coco/images/*/config.json`
3. Create your own captions preprocessing config and put it in the directory `datasets/coco/annotations/your_preprocessing_name/config.json` or use the existing ones at `datasets/coco/annotations/*/config.json`
4. Run script `src/preprocess.py --split_annotations --annotations_processing_config path_to_config --images_processing_config path_to_config`

To run model training on a preprocessed dataset do the following:
1. Create a model config and put it in the directory `models/model_name/config.json` or use the existing ones at `models/*/config.json`
2. Run script `src/run.py --config path_to_config`

## Acknowledgements
The repository structure is inspired by [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) and [How to plan and execute your ML and DL projects](https://blog.floydhub.com/structuring-and-planning-your-machine-learning-project/) articles.
