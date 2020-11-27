# image-captioning
My PhD studies on the image captioning topic

Useful links:
[Data format](https://cocodataset.org/#format-data)
[Data download](https://cocodataset.org/#download)
[Data download instruction](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9)

Notes about project structure:
[Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
[How to plan and execute your ML and DL projects](https://blog.floydhub.com/structuring-and-planning-your-machine-learning-project/)

Заметки на полях:
* импортировать Python модули из git submodule как-то нетривиально...
* если скачать файл с аннотациями с [официального сайта COCO](https://cocodataset.org/#download), то используя что API из документации, что [официальный пакет](https://github.com/cocodataset/cocoapi) для чтения (и более продвинутый [неофициальный](https://github.com/ruotianluo/coco-caption), мы получаем ошибку. Судя по всему, формат файла какой-то неправильный. Поэтому хоть такого я нигде и не видел, попробуем заиспользовать [реализацию из PyTorch](https://pytorch.org/docs/stable/torchvision/datasets.html#captions).

TODO:
* предобработка датасета заранее: чистка, токенизация (PTBTokenizer?), сохранение словаря
* создание своего класса датасета и даталоадера на основе текущих (эти не подходят из-за костылей + невозможности сохранить эмбединги заранее)

TODO (опционально):
* загрузка картинок в tensorboard
