# Файли моделей із `prep-main-pipeline-X.ipynb`

Покладіть класифікаційні checkpoint-файли в цю папку **під їхніми оригінальними назвами**:

| Вхід | Файли ваг |
| --- | --- |
| Вихідне фото | `efficientnet_original.weights.h5`, `deit_original.weights.h5`, `vit_original.weights.h5`, `swin_original.weights.h5` |
| Crop після DeepLab | `efficientnet_segmented_deeplab.weights.h5`, `deit_segmented_deeplab.weights.h5`, `vit_segmented_deeplab.weights.h5`, `swin_segmented_deeplab.weights.h5` |
| Ablation після Otsu | `deit_ablation_otsu.weights.h5`, `swin_ablation_otsu.weights.h5` |
| Ablation після Grad-CAM | `deit_ablation_gradcam.weights.h5`, `swin_ablation_gradcam.weights.h5` |
| Ablation після U-Net | `deit_ablation_unet.weights.h5`, `swin_ablation_unet.weights.h5` |
| Ablation після SegNet | `deit_ablation_segnet.weights.h5`, `swin_ablation_segnet.weights.h5` |

Ноутбук також міг зберегти повні моделі у форматі `*.keras`, але при повторному використанні готового checkpoint він пропускає `model.save()`. Тому для інтеграції надійніше орієнтуватися на `*.weights.h5`: спочатку відтворити архітектуру з `build_classifier()`, потім викликати `load_weights()`. Це **не** файли для прямого `load_model()`.

Сегментаційні checkpoint-файли кладіть у [сусідню папку](../segmentation_models/README.md): `segmenter_unet.weights.h5`, `segmenter_segnet.weights.h5`, `segmenter_deeplab.weights.h5`. Otsu не потребує ваг. Grad-CAM використовує `efficientnet_original.weights.h5`, а не окремий сегментатор.

У фінальному шляху ноутбука сегментація DeepLab **не вбудована** у класифікатори `*_segmented_deeplab.weights.h5`. Послідовність у сайті така: RGB-фото → видалення волосся → DeepLab-маска → найбільша зв'язна область маски → crop із відступом 8% → resize до 224×224 → відповідний класифікатор. Класифікатори `*_original.weights.h5` отримують очищене RGB-фото без crop. Новий ноутбук створює очищені від волосся копії за допомогою Black Hat та inpainting, але має `REMOVE_HAIR=False`. Сайт завжди застосовує видалення волосся перед сегментацією та класифікацією, включно зі звичайним режимом; тому вхід відрізняється від зображень, на яких навчалися наявні checkpoint-и.

Важливі параметри інференсу: класифікатори отримують 224×224; сегментатори U-Net, SegNet і DeepLab — 256×256; порядок класів із `LabelEncoder`: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`. DeiT нормалізує RGB за ImageNet mean/std, інші класифікатори отримують RGB у діапазоні 0–255 і використовують внутрішнє масштабування. Після порогування маски використовується `> 0.5`; для U-Net береться канал 0, для SegNet і DeepLab — канал 1.

Старий гостьовий маршрут окремо використовує `resnet_model.h5`. Файли ваг ігноруються Git; Docker Compose монтує ці папки без включення ваг до образу.

Дослідницький API завантажує ці ваги через `app/experiments/inference.py`. Для звичайної пари Swin + DeepLab використовується фінальний `swin_segmented_deeplab.weights.h5`; додатковий DeepLab ablation не використовується. Ablation-файли Otsu, Grad-CAM, U-Net і SegNet використовуються в Research для класифікації відповідних масок, коли вони наявні. XAI обчислюється з цих самих класифікаторів у `app/experiments/xai.py`.
