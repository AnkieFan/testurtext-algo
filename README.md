# Testurtext 测测你写的像谁
This is the repository of algorithms behind the website testurtext.us (and testurtext.site). This website tests which writers you write like (in Chinese). It not only identifies the most similar authors but also shows users which sentences led to that conclusion.  

### Algorithms
+ Classifier: [FastText](https://fasttext.cc/)
+ Explainable AI: [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/shap/shap)

## Model Training 
### Data Preparation 数据准备
1. 准备你的训练数据集
2. 更改[data_processor.py](model/data_processor.py)中路径设置
3. Run `python model/data_processor.py`
4. 如果需要前端展示：手动补充更新[author.csv](model/author.csv)，ID即为fasttext中label

### Train your own model 训练你的模型
Run `python model/train.py`

## License
Copyright (C) 2024 Ankie Fan at Department of Advanced Computing Sciences, Faculty of Science and Engineering, Maastricht University

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for more information.

# Contact
If you have any questions or suggestions, feel free to leave in [Discussions](https://github.com/AnkieFan/testurtext-algo/discussions)
