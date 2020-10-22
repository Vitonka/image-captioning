# image-captioning
My PhD studies on the image captioning topic

Useful links:
[https://cocodataset.org/#format-data](Data format)
[https://cocodataset.org/#download](Data download)
[https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9](Data download instruction)

Notes about project structure:
[https://drivendata.github.io/cookiecutter-data-science/](Cookiecutter Data Science)
[https://blog.floydhub.com/structuring-and-planning-your-machine-learning-project/](How to plan and execute your ML and DL projects)


Заметки на полях:
* импортировать Python модули из git submodule как-то нетривиально...
* если скачать файл с аннотациями с [https://cocodataset.org/#download](официального сайта COCO), то используя что API из документации, что [https://github.com/cocodataset/cocoapi](официальный пакет) для чтения (и более продвинутый [https://github.com/ruotianluo/coco-caption](неофициальный), мы получаем ошибку. Судя по всему, формат файла какой-то неправильный. Поэтому хоть такого я нигде и не видел, попробуем заиспользовать [https://pytorch.org/docs/stable/torchvision/datasets.html#captions](реализацию из PyTorch)
