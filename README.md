# ------
## Обзор решения

Данный обзор отражает последовательность решений - файлов и ноутбуков. Проверять в этом порядке.

1. ```research.ipynb``` содержит небольшое исследование данных и предобработку, необходимую для относительно легких моделей.
2. ```lite_model.ipynb``` содержит проверку легких моделей.
3. ```Sequential_NN.py``` содержит проверку простой нейронной сети.
4. ```BERT_NLP_Contur_FakeNews.ipynb``` содержит проверку RuBERT (вычисления производились в Google Colab).
5. ```resources.txt``` - дополнительный файл, нет необходимости проверять, но если возникнут вопросы, то можно заглянуть.

Подробнее расписано в каждом из файлов.

Для файла ```predictions.tsv``` была использована последняя модель - RuBERT. Обученная модель, использованная для предсказания находится на диске:
[RuBERT_from_SKOLKOVO](https://drive.google.com/file/d/1D6f61XKVAbQCOE0QQWcmxkPNL1SDRRIv/view?usp=sharing)