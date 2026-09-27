# Skin Insight AI — бекенд

FastAPI API для класифікації зображень шкіри, XAI-пояснень та історії користувачів. Дані й зображення зберігаються в MongoDB/GridFS.

## Запуск через Docker Compose

Потрібні Docker і Docker Compose. Для гостьового аналізу потрібна модель `app/classification_models/resnet_model.h5`; без неї цей старий маршрут повертає `503`. Для авторизованого аналізу потрібні checkpoint-файли з [переліку моделей](app/classification_models/MODELS.md). Файли `.h5` не входять до Git-репозиторію.

З каталогу бекенду виконайте:

```bash
docker compose up --build
```

Якщо локальний порт `8000` зайнятий, використайте `API_PORT=8001 docker compose up --build`.

API буде доступне на <http://localhost:8000>, документація OpenAPI — на <http://localhost:8000/docs>. Compose запускає API та MongoDB; база зберігається в Docker volume `mongo_data`. Папки з вагами монтуються в API-контейнер без копіювання багатогігабайтних файлів в образ. Під час першого збирання завантажуються великі ML-залежності, тому воно може тривати кілька хвилин. API збирається для `linux/amd64`; на Apple Silicon Docker використовуватиме емуляцію.

Для локального PoC Compose задає `SECRET_KEY` за замовчуванням. За потреби передайте власний:

```bash
SECRET_KEY=your-local-secret docker compose up --build
```

Зупинити стек: `docker compose down`. Видалити також дані MongoDB: `docker compose down -v`.

## Запуск разом із фронтендом

Якщо обидва репозиторії лежать поруч у каталозі з назвами `skin-insight-thesis-backend` і `skin-insight-thesis-frontend`, можна запустити весь стек однією командою з каталогу фронтенду:

```bash
cd ../skin-insight-thesis-frontend
docker compose up --build
```

Фронтенд буде на <http://localhost:3000>. Не запускайте обидва Compose-стеки одночасно: вони використовують однаковий порт API `8000`.

## Новий дослідницький API

Маршрути `/api/diagnostics/analyze` та `/api/diagnostics/explain` доступні лише з Bearer-токеном зареєстрованого користувача. Класифікація, сегментація й XAI обчислюються з реальних checkpoint-ів і повертають `demo: false`. Старий префікс `/api/experiments` тимчасово залишено для окремого референсного вікна Research.

- `GET /api/diagnostics/capabilities`: повертає доступні пари класифікатора та сегментації за наявними файлами ваг і підтримувані XAI-методи для кожного класифікатора.

- `POST /api/diagnostics/analyze`: `multipart/form-data` з `file`, `models` (JSON-масив: `efficientnetb0`, `vit`, `deit`, `swin`), `segmentation_enabled` (`true` або `false`) та `compare_segmentations` (`true` за замовчуванням). Якщо сегментацію ввімкнено, API запускає DeepLabV3+, Otsu, Grad-CAM, U-Net і SegNet для порівняння масок і класифікує всі пари, для яких є ваги; DeepLabV3+ йде першим. Якщо вимкнено, використовується фото без сегментації та ваги `*_original.weights.h5`. Basic передає `compare_segmentations=false` і використовує лише DeepLabV3+. Відповідь містить `run_id`, `demo`, `segmentations`, `remove_hair_artifacts`, `results` (по одному на доступну пару класифікатора й сегментації; `configuration_id`, `model_id`, `segmentation_id`, сім імовірностей), `unavailable_configurations` та `stages_by_segmentation` (PNG у base64: `original`, `hair_removed`, `mask`, `overlay`, `roi`). Волосся завжди видаляється методом Black Hat + inpainting.
- `POST /api/diagnostics/explain`: JSON з `run_id`, `configurations` (масив `configuration_id`, наприклад `["swin:deeplabv3plus"]`) і `methods` (`attention_rollout`, `transformer_attribution`, `transition_attention_maps`, `integrated_gradients`, `occlusion_sensitivity`, `gradcam`, `swin_transformer_attribution`). Відповідь містить `demo: false`, масив `explanations` із `configuration_id`, `model_id`, `segmentation_id`, `method`, `predicted_class`, `overlay_image`, `heatmap_image` (PNG у base64) та `unavailable_explanations` для несумісних пар. Для запуску з однією сегментацією також приймається старий параметр `models` замість `configurations`.

Attention Rollout, Transformer Attribution і Transition Attention Maps доступні для ViT та DeiT. Grad-CAM доступний для EfficientNetB0. Метод із ноутбука `Swin Transformer Attribution` доступний для Swin; технічно це додатна частина градієнта за входом × пікселі, а не увага вікон Swin. Integrated Gradients і Occlusion Sensitivity доступні для всіх моделей. XAI використовує той самий 224×224 crop, що й класифікація відповідної конфігурації. Integrated Gradients використовує 64 кроки від нульового входу; Occlusion Sensitivity приховує квадрати 32×32 із кроком 16 та вимірює падіння ймовірності передбаченого класу. Обчислення XAI для кількох великих моделей може тривати довго на CPU.

Запуски тимчасово зберігаються в пам’яті процесу (до 20); після перезапуску API попередній `run_id` недійсний. Архітектури й обробка з ноутбука перенесені в `app/experiments/inference.py`; класифікатор і сегментатор завантажуються на вимогу, тому перший запуск може бути тривалим. KerasHub під час першого створення ViT/DeiT/Swin/DeepLab може завантажити невеликі файли конфігурації preset; потрібен доступ до мережі.
