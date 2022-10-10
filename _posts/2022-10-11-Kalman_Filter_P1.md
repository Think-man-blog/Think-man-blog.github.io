---
layout: post
author: nguyenthanhminh
title: Algorithm 1 - Kalman Filter Part 1
---

# 1. Đôi nét về Kalman Filter

## 1.1 Giới thiệu chung
Bạn có biết rằng Kalman Filter được sử dụng để dẫn đường cho
chiến dịch Apolo vào năm 1960. Vậy bộ lọc này phải được chứng minh
là đủ manh mẽ để được áp dựng vào trong những chuyến hành trình
trong không gian như vậy.

<img src="_posts\Kalman_Filter\Kalman_Filter_thumbnail.jpg" width="600px" hieght="300px">

***Đầu tiên mình sẽ tóm gọn lại Kalman Filter là gì:***

Kalman Filter thực chất là một phương pháp dự đoán trạng thái tiếp theo của object, và dựa vào quan sát của hệ thống để có thê tinh chỉnh lại các thông số, tăng độ tin cây cho lần dự đoán tiếp theo. Ví dụ để dễ hiểu hơn: Khi bạn chơi bóng đá, ban sẽ dự đoán vị trí tiếp theo của quả bóng để đỡ đúng không nào ️⚽ (đây gọi là dự đoán), nhưng đột
nhiên banh lại đi lệch một chút so với dự đoán của bạn 💨⚽ (đây gọi là quan sát) vậy ở lần dự đoán tiếp theo bạn sẽ sử dụng thông tin của quả bóng bị lệch hướng để có được dự đoán chính xác cao. Dễ hiểu phải không nào. Vậy thì làm sao mà Kalman Filter làm được như thế, hãy cùng mình
đi sâu vào bài toán nhé 🤓🤓🤓.

## 1.2 Kalman Filter
Chúng ta bắt đầu với dự đoán trong Kalman Filter.
Giả sử ta có vector chứa thông tin về vị trí và vận tốc của quả bóng: 

$$ \mathbf{x} = \begin{bmatrix} 
position \\
velocity
\end{bmatrix} $$ 

hay tổng quát hơn $\mathbf{x} \in \mathbb{R}^{n}$ chứa n thông tin của $\mathbf{x}$ cần dự đoán. 

Ma trận $ \mathbf{F} \in \mathbb{R}^{n*n}$ dùng tổ hợp tuyến tính của $\mathbf{n}$ thông tin từ $\mathbf{x}$ để dự đoán tráng thái (state) tiếp theo được gọi là mô hình chuyển đổi trạng thái (The state-transition model). 

Ví dụ ta xem như quả bóng chuyển động đều, dễ thấy rằng: $position^{t+1} = position^{t} + velocity$ giả sử $t = 1$. Lúc này ma trận $ \mathcal{F} $ của chúng ta sẽ là: 

$$ \mathbf{F} = \begin{bmatrix}
1 & 1\\ 
0 & 1
\end{bmatrix} $$ 

Và (lưu ý vì là chuyển động đều nên vận tốc là hằng):

$$\mathbf{x}^{t+1} = \mathbf{F}*\mathbf{x}^{t}= \begin{bmatrix}
1&1\\ 
0&1
\end{bmatrix} 
* 
\begin{bmatrix}
position^{t} \\
velocity^{t}
\end{bmatrix} \\ 

= \begin{bmatrix}
position^{t} + velocity^{t}\\
velocity^{t}
\end{bmatrix}
= \begin{bmatrix}
position^{t + 1}\\
velocity
\end{bmatrix}
$$ 

Tiếp theo đến quan sát $\mathbf{y} \in \mathbb{R}^{m}$, ở đây ví dụ như thứ ta quan sát được là vị trị của quả bóng chẳng hạn. 

$$ \mathbf{y} = \begin{bmatrix}
position \\
\end{bmatrix} $$ 

***Các bạn sẽ tự hỏi tại sao số chiều của quan sát (observer) và dự đoán (estimator) lại khác nhau. Vì mô hình muốn quan sát những thông tin mà các hệ thống không thể quan sát được và thực tế thông tin quan sát được ít hơn so với những gì muốn đo lường.***

Và ma trận $ \mathbf{H} \in \mathbb{R}^{m*n}$ được gọi là mô hình quan sát (The observation model), chuyển đổi từ không gian dự đoán sang không gian quan sát, hay đơn giản hơn là sử dụng các tổ hợp tuyến tính thông tin của dự đoán $ \mathbf{x} $ sang thành quan sát $ \mathbf{y} $. 

$$ \mathbf{y}^{t} = \mathbf{H}*\mathbf{x}^{t} 
= 
\begin{bmatrix}
1\\ 
0
\end{bmatrix} 
*
\begin{bmatrix}
position^{t} \\
velocity^{t}
\end{bmatrix} \\\\ 
= \begin{bmatrix}
position^{t} 
\end{bmatrix} \\ 
$$

Tuy nhiên trong thực tế chẳng bao giờ mọi chuyện xảy ra suôn sẻ như vậy cả, chúng ta luôn gặp các giá trị nhiễu trong quá trình quan sát lẫn dự đoán, vậy nên Kalman Filter đã giả sử các tham số nhiễu như sau:

$$\mathbf{x}^{t+1} = \mathbf{F}*\mathbf{x}^{t} + \mathbf{w}^{t}$$

$$\mathbf{y}^{t+1} = \mathbf{H}*\mathbf{x}^{t} + \mathbf{v}^{t}$$

Với $\mathbf{w} \thicksim \mathcal{N}(0, \mathbf{Q}) \in \mathbb{R}^{n}$, và $\mathbf{v} \thicksim \mathcal{N}(0, \mathbf{R}) \in \mathbb{R}^{m}$, yếu tố $t$ đối với nhiễu có thể bỏ qua và xem $\mathbf{w}$ và $\mathbf{v}$ là 2 $\mathcal{vector}$ ngẫu nhiên.

# 2. Các bước thực hiện của thuật toán Kalman Filter

Phía trên là giới thiệu sơ lược về mô hình Kalman Filter cơ bản, có thể tóm gọn lại như sau:

$\mathbf{x}\in \mathbb{R}^{n}$ là trạng thái của object có kèm theo yếu tố thời gian.

$\mathbf{y}\in \mathbb{R}^{m}$ là quan sát của object kèm theo yếu tố thời gian.

$ \mathbf{F} \in \mathbb{R}^{n*n}$ là mô hình chuyển đổi trạng thái (The state-transition model). 

$ \mathbf{H} \in \mathbb{R}^{m*n}$ là mô hình quan sát (The observation model).

$\mathbf{w} \thicksim \mathcal{N}(0, \mathbf{Q}) \in \mathbb{R}^{n}$ là nhiễu của dự đoán.

$\mathbf{v} \thicksim \mathcal{N}(0, \mathbf{R}) \in \mathbb{R}^{m}$ là nhiễu của quan sát.

Gọi $\Sigma_{x}$ là ma trận covariance của x kèm yếu tố thời gian, và kỳ vọng là $\overline{x}$. Giả sử ta xem $\mathbf{x} \thicksim \mathcal{N}(\overline{x}, \Sigma_{x})$

***Các bước của thuật toán Kalman Filter***

***Dự đoán:***

$$\overline{\mathbf{x}^{t+1}} = \mathbf{F}*\overline{\mathbf{x}^{t}} \hspace{1cm} (1)$$ 

$$ \Sigma_{x}^{t+1} = \mathbf{F}*\Sigma_{x}^{t}*\mathbf{F}^{\mathbf{T}} + \mathbf{Q} \hspace{1cm} (2)$$ 

$(1)$ và $(2)$ là công thức cập nhật bình thường, tính toán kỳ vọng và ma trận $covariance$ của dự đoán dựa trên tổ hợp tuyến tính của các $vector$ ngẫu nhiên. Vì bài toán này cũng không quá khó, nên mình sẽ không chứng minh lại phần này. Bạn có thể tìm hiểu với từ khóa Linear Combination With Random Vector hoặc ở [đây](http://www.math.kent.edu/~reichel/courses/monte.carlo/alt4.7d.pdf).

***Cập nhật:***

Đầu tiên ta tính toán hệ số $Kalman$ tại thời điểm $t+1$, $t+1$ lúc này là thời điểm mà bạn dự đoán quả bóng ⚽ sẽ di chuyển như thế nào trong tương lai ấy.

$$\mathbf{K}^{t+1} = \Sigma_{x}^{t+1}*\mathbf{H}^{\mathbf{T}} (\mathbf{R} + \mathbf{H}*\Sigma_{x}^{t+1}*\mathbf{H}^{\mathbf{T}})^{-1}  \hspace{1cm} (3)$$ 

Sau đó bạn sẽ cập nhật lại các giá trị kỳ vọng và ma trận $covariance$ dựa vào hệ số $Kalman$ và quan sát tại thời điểm $t+1$, lúc này thời điểm $t+1$ đã xảy ra bạn mới quan sát được đúng chứ? Bạn còn nhớ quả bóng bị gió thổi bay 💨⚽ ở đầu bài viết không? Lúc này ta sẽ dùng quan sát này kèm với dự đoán lúc trước của bản thân để giúp cho lần dự đoán tiếp theo chính xác hơn.

$$\overline{\mathbf{x}^{t+1}}' = \overline{\mathbf{x}^{t+1}} + \mathbf{K}^{t+1}*(\mathbf{y}^{t+1} - \mathbf{H}\overline{\mathbf{x}^{t+1}}) \hspace{1cm} (4)$$ 

$$ {\Sigma_{x}^{t+1}}' = \Sigma_{x}^{t+1}*(I - \mathbf{K}^{t+1}\mathbf{H}^{t+1}) \hspace{1cm} (5)$$ 

Lúc này kỳ vọng tại thời điểm $t+1$ sẽ được cập nhật lại, cụ thể là:

$$ \overline{\mathbf{x}^{t+1}}:=\overline{\mathbf{x}^{t+1}}' $$

$$ {\Sigma_{x}^{t+1}}:={\Sigma_{x}^{t+1}}' $$

Bạn sẽ tự hỏi $\overline{\mathbf{x}^{t+1}}'$ và $ {\Sigma_{x}^{t+1}}'$ làm sao sẽ giúp cho dự đoán tiếp theo chính xác hơn phải không, và hệ số $Kalman$ ở đâu ra. Mình hi vọng có thể giải thích và chứng minh rõ cho bạn ở ***phần 2*** của bài viết, hãy đón chờ nhé.

# 3. Tài liệu

1. [Kalman Filter - Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)

2. [Kalman Filter - Thetalog](https://thetalog.com/machine-learning/kalman-filter/)

3. [Kalman Filter - Viblo](https://viblo.asia/p/sort-deep-sort-mot-goc-nhin-ve-object-tracking-phan-1-Az45bPooZxY#_32-bo-loc-kalman-kalman-filter-9)


