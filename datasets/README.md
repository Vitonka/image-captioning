[Data format](https://cocodataset.org/#format-data)

[Data download](https://cocodataset.org/#download)

[Data download instruction](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9)

Если скачать файл с аннотациями с [официального сайта COCO](https://cocodataset.org/#download), то используя что API из документации, что [официальный пакет](https://github.com/cocodataset/cocoapi) для чтения (и более продвинутый [неофициальный](https://github.com/ruotianluo/coco-caption), мы получаем ошибку. Судя по всему, формат файла какой-то неправильный. Поэтому хоть такого я нигде и не видел, попробуем заиспользовать [реализацию из PyTorch](https://pytorch.org/docs/stable/torchvision/datasets.html#captions).
