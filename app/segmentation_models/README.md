# Ваги сегментації з `main_pipeline (2).ipynb`

Покладіть сюди `segmenter_unet.weights.h5`, `segmenter_segnet.weights.h5` і `segmenter_deeplab.weights.h5`. Це checkpoint-файли ваг: для `load_weights()` спочатку потрібно створити архітектури `build_unet()`, `build_segnet()` і `build_deeplab()` із ноутбука.

Otsu не потребує файлу. Grad-CAM використовує вихідний класифікатор EfficientNet із `../classification_models/efficientnet_original.weights.h5`.

Дослідницький API завантажує ці checkpoint-файли для побудови реальних масок. Для сегментації потрібна архітектура з ноутбука та сумісна версія KerasHub.
