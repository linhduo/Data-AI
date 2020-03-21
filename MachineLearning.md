---
title: machine learning
layout: default
---

# Machine Learning

Machine learning ("học máy" hay "máy học") là một nhánh của AI (trí tuệ nhân tạo), chuyên xây dựng các thuật toán để máy tính __tự học__ các qui luật và tri thức từ các tập __dữ liệu__ được cung cấp. Nó khác với cách tiếp cận cổ điển của AI, gọi là symbolic AI, ở đó các tri thức và quy luật được người dùng lập trình cứng (hard code) cho máy tính thực hiện.

Có ba loại machine learning phổ biến: supervised learning (học giám sát), unsupervised learning (học không giám sát), và reinforcement learning (học tăng cường). Trong supervised learning các mẫu dữ liệu được đánh nhãn để máy tính (thuật toán) dựa vào đấy quan sát và đưa ra quy luật. Trong unsupervised learning, các dữ liệu không có nhãn. Trong reinforcement learning thuật toán (agent) học thông qua việc tương tác với môi trường (và nhận lại các "phần thưởng" từ các tương tác đó).  

__Supervised learning__ Có hai loại cơ bản: classification (phân loại hay phân lớp) và regression (hồi quy).

- Classification: tập hợp các giá trị nhãn là hữu hạn và rời rạc. Ví dụ: phân loại động vật trong các bức ảnh (chó, mèo, etc), phân loại học sinh (xuất sắc, giỏi, khá, TB, yếu), phân loại nguy cơ tử vong của bệnh nhân (sống, chết).

- Regression: tập hợp các giá trị nhãn là liên tục. Ví dụ: giá tiền mua nhà,  

__Unsupervised learning__ Có khá nhiều các thể loại: clustering (phân cụm), dimension reduction, knowledge/data representation, etc. 

- Clustering: được dùng trong việc khám phá những nhóm dữ liệu có mối tương đồng. Ví dụ: nhóm những người có liên hệ mật thiết với nhau trên mạng xã hội, những nhóm hàng có tính chất tương tự, etc.

- Dimension reduction: tìm kiếm những tính chất quan trọng của dữ liệu (lược bỏ những cái không quan trọng).

- Knowledge/data representation: biểu diễn tri thức bằng các cấu trúc mà máy tính có thể sử dụng hiệu quả. Tác vụ này đôi khi là bước tiền xử lý cho các thuật toán machine learning khác. 

__Reinforcement learning__ nghiên cứu các thuật toán để dạy agent tương tác với môi trường xung quanh (thực thi một tác vụ nào đó). Được ứng dụng nhiều trong điều kiển robot và game.

Các loại machine learning khác:

- Self-supervised, semi-supervised learning: các biến thể kết hợp giữa supervised và unsupervised learning. 


# Machine Learning in Python

Python đang là ngôn ngữ chính cho machine learning (ML). Nó khá tiện dụng cho cả việc quan sát (visualize), khám phá (explore), dữ liệu lẫn xây dụng mô hình (model) cho ML. Một số công cụ cơ bản trong Python cho ML:

1. Pandas: dùng để quan sát và khám phá dữ liệu bảng (tabular data) và dữ liệu theo thời gian (time series data).

2. Matplotlib và Opencv: dùng để quan sát và khám phá dữ liệu hình ảnh.

3. Sklearn: dùng để tạo các mô hình ML một cách nhanh chóng. Nó được dùng như các công cụ hộp đen (blackbox) khi người dùng muốn có kết quả mà k cần phải bỏ nhiều thời gian xây dựng thuật toán. 

4. Tensorflow, Keras, Pytorch là các platform dùng để xây dựng các mô hình ML phức tạp và có thể tinh chỉnh. Phù hợp với những người muốn đi sâu vào thuật toán. Keras hiện nay đã được nhập vào Tensorflow thành tf.Keras.

Một số tác vụ cơ bản trong ML:

- Quan sát và khám phá dữ liệu: trước khi xây dựng mô hình ML thì cần phải hiểu là dữ liệu nó nhìn như thế nào, có những đặc tính cơ bản gì. 

- Xây dựng luồng dữ liệu để huấn luyện mô hình: tác vụ này đôi khi khá cơ bản. Chỉ đơn giản là chia dữ liệu thành ba tập: tập train, tập validate, và tập test. Tuy nhiên đối với dữ liệu quá lớn và phức tạp (từ nhiều nguồn) thì tác vụ này rất phức tạp. Đây là công việc của một Data Engineer.

- Xây dựng mô hình và huấn luyện: xác định loại mô hình phù hợp (decision tree, SVM, deep learning, etc), sau đó xây dựng mô hình tương ứng, và huấn luyện mô hình (model training). Đây là công việc của một Data Scientist. 

- Đưa mô hình vào ứng dụng: tác vụ này khá phức tạp và có thể được chuyên trách bởi một AI Engineer. 

# Một số tập dữ liệu đơn giản

Để học tập/nghiên cứu về ML, người học cần phải làm quen với một số bộ dữ liệu cơ bản.
- Dữ liệu dạng bảng: Iris Flower, Titanic Survival, Boston Housing
- Dữ liệu dạng hình ảnh: MNIST, CIFAR-10, CIFAR-100
- Dữ liệu dạng text: IMDB Movie Review Sentiment Classification (Stanford), Reuters Newswire Topic Classification (Reuters-21578), News Group Movie Review Sentiment Classification (Cornell), Twitter Sentiment Anlysis (Kaggle)

# Ví dụ:  tập dữ liệu Iris Flower
Chúng ta sẽ tìm hiểu tập dữ liệu "Iris Flower" (hoa Iris). Hoa Iris có ba loại: Iris setosa, Iris virginica, và Iris versicolor.
![width=5cm](iris.png)
Việc phân loại hoa thường dựa vào kích thước của cánh hoa lớn (senpal) và cánh nhỏ (petal). Bộ dữ liệu này thường được sử dụng để kiểm tra độ chính xác của các thuật toán ML. Nó có thể được load trực tiếp từ sklearn. 


```python
from sklearn.datasets import load_iris
iris = load_iris()
```

Việc đọc dữ liệu để quan sát khá đơn giản (dùng pandas).


```python
import pandas as pd
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target,columns = ['types'])

#Kết nối dữ liệu và nhãn
label = pd.concat([X,y],axis=1)
label
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>types</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>




Ở trên nhãn được ký hiệu bằng con số, không thực sự trực quan. Có một cách để đổi nó sang dữ liệu thể loại (Categorical data) có tên rõ ràng.


```python
y = pd.Categorical.from_codes(iris.target, iris.target_names)
lab = pd.DataFrame(y[:],columns =['type'])
lab.head()
new_label = pd.concat([X,lab],axis=1)
new_label.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>



Tuy nhiên, khi làm ML, người ta hay chuyển các label sang "one-hot-vector". Cái này rất sẽ làm trong Pandas, bằng cách gọi lệnh dummies.


```python
# Lưu ý, kết quả là một DataFrame trong pandas.
y=pd.get_dummies(y)
y.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>setosa</th>
      <th>versicolor</th>
      <th>virginica</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


Chia dữ liệu thành tập train và tập test. Cái này có thể làm bằng việc gọi sklearn.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

Thường tập train này sẽ còn được chia làm hai phần nhỏ. Một phần dùng cho việc training, một phần cho validation. Hai khái niệm này sẽ được làm rõ dần trong những bài sau.

# Kết luận và gợi ý

Đây là bài viết giới thiệu cho series cơ bản "Machine Learning in Python". Series sẽ được viết bằng tiếng Việt (có sử dụng một số thuật ngữ tiếng Anh); bao gồm các bài viết mang tính gợi mở, giúp người đọc có định hướng trong việc học tập.

## Bài tập:
1. Liệt kê các bộ dữ liệu cơ bản trong sklearn.
2. Nêu tên một số bộ dữ liệu nổi tiếng khác (không liệt kê trong bài viết). Giới thiệu sơ lược về các bộ dữ liệu đó 
3. Download, quan sát, và khám phá bộ dữ liệu Titanic Survival. 


## Tài liệu tham khảo:
1. [sklearn](https://github.com/scikit-learn/scikit-learn)
2. [padas](https://pandas.pydata.org)
3. [Trang chuyên tìm kiếm về dataset của Google](https://datasetsearch.research.google.com)
4. Tài liệu tốt để học ML với Python[Python Machine Learning](https://sebastianraschka.com/books.html) của Sebastian Raschka.
