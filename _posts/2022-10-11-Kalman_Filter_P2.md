---
layout: post
author: nguyenthanhminh
title: Algorithm - Kalman Filter Part 2
---

# 1. Nhắc lại về Kalman Filter

Mình sẽ tóm tắt Kalman Filter dưới dạng các ký hiệu toán học và không giải thích như phần 1 nữa, các bạn nếu chưa hiểu rõ có thể quay lại [phần 1](https://think-man-blog.github.io/2022/10/09/Kalman_Filter_P1.html).

Kalman Filter gồm các thành phần như sau:

$\mathbf{x}\in \mathbb{R}^{n}$ là trạng thái của object có kèm theo yếu tố thời gian.

$ \mathbf{F} \in \mathbb{R}^{n*n}$ là mô hình chuyển đổi trạng thái (The state-transition model). 

$$\mathbf{x}^{t+1} = \mathbf{F}*\mathbf{x}^{t} + \mathbf{w}^{t} \hspace{1cm} (1)$$ 

$\mathbf{w} \thicksim \mathcal{N}(0, \mathbf{Q}) \in \mathbb{R}^{n}$ là nhiễu của dự đoán.

Ta xem $\mathbf{x} \thicksim \mathcal{N}(\overline{x}, \Sigma_{x})$. Gọi $\Sigma_{x}$ là ma trận covariance của x kèm yếu tố thời gian, và kỳ vọng là $\overline{x}$. 

$\mathbf{y}\in \mathbb{R}^{m}$ là quan sát của object kèm theo yếu tố thời gian.

$ \mathbf{H} \in \mathbb{R}^{m*n}$ là mô hình quan sát (The observation model).

$$\mathbf{y}^{t} = \mathbf{H}*\mathbf{x}^{t} + \mathbf{v}^{t} \hspace{1cm} (2)$$

$\mathbf{v} \thicksim \mathcal{N}(0, \mathbf{R}) \in \mathbb{R}^{m}$ là nhiễu của quan sát.

Ta xem $\mathbf{y} \thicksim \mathcal{N}(\overline{y}, \Sigma_{y})$. Gọi $\Sigma_{y}$ là ma trận covariance của x kèm yếu tố thời gian, và kỳ vọng là $\overline{y}$. 

Diễn giải dưới đây được dẫn từ [Thetalog](https://thetalog.com/machine-learning/kalman-filter/), mình khuyên các bạn cũng nên đọc Thetalog để có thể hiểu rõ hơn về Kalman Filter.

Ta có giả định:

$$\mathbf{y}^{t} = \mathbf{H}*\mathbf{x}^{t} + \mathbf{v}^{t} $$

Nhưng việc biết $\mathbf{y}^{t}$ sẽ giúp gì cho lần cập nhật tiếp theo? Xét biểu thức:

$$p(x^{t}|y^{t}) = \frac{p(y^{t}|x^{t})p(x^{t})}{p(y^{t})} \hspace{1cm} (3)$$

Trong đó:

$p(x^{t}|y^{t})$ 
là hàm phân bố xác suất của 
$\mathbf{x}^{t}$ 
khi biết 
$\mathbf{y}^{t}$

$p(y^{t}|x^{t})$ 
là hàm phân bố xác suất của 
$\mathbf{y}^{t}$ 
khi biết 
$\mathbf{x}^{t}$ 


$p(y^{t})$ và 
$p(x^{t})$ lần lượt là hàm phân bố xác suất của 
$\mathbf{y}^{t}$ và 
$\mathbf{x}^{t}$ 

Lúc này việc có thông tin về 
$\mathbf{y}^{t}$ 
sẽ giúp chúng ta cập nhật lại hàm phân bố xác suất cho 
$\mathbf{x}^{t}$ thông qua 
$p(x^{t}|y^{t}).$ Từ đó những quan sát tiếp theo sẽ mang độ chính xác cao hơn.

Các blog hiện nay đều bỏ qua đi phần chứng minh này, nhưng mình sẽ chứng minh từng bước cho bạn thấy từ đâu mà hệ số $Kalman$ xuất hiện. 

# 2. Hệ số Kalman

Lưu ý từ đây sẽ nồng nặc mùi toán học, hãy đọc chậm rãi và cẩn thận nhé. Chúng ta sẽ chứng minh các bài toán nhỏ để ra được hệ số Kalman hoàn thiện nhé.

## 2.1 Schur Complement of matrix

Cụ thể "Schur Complement of matrix" biểu diễn ma trận $\mathbf{M}^{-1}$ dựa trên 4 khối ma trận nằm trong ma trận $\mathbf{M}$ vuông.

Xét ma trận $\mathbf{M}$ gồm $2 \times 2$ các khối $matrix$:

$$\mathbf{M} = 
\begin{bmatrix}
\mathbf{A} & \mathbf{B} \\
\mathbf{B}^\mathsf{T} & \mathbf{D}
\end{bmatrix} \hspace{1cm} (4)$$

Với $\mathbf{A} \in \mathbb{R}^{p \times p}$, $\mathbf{B} \in \mathbb{R}^{p \times q}$ và $\mathbf{D} \in \mathbb{R}^{q \times q}$. Có thể suy ra kích thước của $\mathbf{M}\in \mathbb{R}^{(p+q) \times (p+q)}$ 

***Ở đây minh sẽ chứng minh cho trường hợp đặt biệt, còn tổng quát bạn hãy thay $\mathbf{B}^{T}$ thành $\mathbf{C}$ là được nhé, cũng có thể xem như là bài tập dành cho bạn để quen với ma trận hơn.***

Để giải bài toán này cũng cực kỳ đơn giản, hẳn là bạn đã từng giải qua hệ phương trình bằng ma trận rồi phải không.

$$\mathbf{M} \times \mathbf{a} = \mathbf{b} \Rightarrow \mathbf{a} = \mathbf{M}^{-1} \times \mathbf{b} \hspace{1cm} (5)$$

***Note***: ở đây $x$, $y$ là các $vector \in \mathbb{R}^{(p+q)}$, và mỗi khi $matrix^{-1}$ ta giả sử $matrix$ khả nghịch

Cho:

$$\mathbf{a} = \begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix};
\space
\mathbf{b} = \begin{bmatrix}
\mathbf{u} \\
\mathbf{v}
\end{bmatrix}
;\space a, u \in \mathbb{R}^{p} \space và \space b, v \in \mathbb{R}^{q}$$

Từ $(4)$ và $(5)$ ta có:

$$\mathbf{M} \times \mathbf{x} = \mathbf{y} \Leftrightarrow
\begin{bmatrix}
\mathbf{A} & \mathbf{B} \\
\mathbf{B}^\mathsf{T} & \mathbf{D}
\end{bmatrix}
\times 
\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix} = 
\begin{bmatrix}
\mathbf{u} \\
\mathbf{v}
\end{bmatrix}
\hspace{1cm} (6)$$

$$\Rightarrow 
\begin{bmatrix}
\mathbf{A} \mathbf{x} + \mathbf{B}y \\
\mathbf{B}^\mathsf{T}\mathbf{x} + \mathbf{D}\mathbf{y}
\end{bmatrix}=
\begin{bmatrix}
\mathbf{u} \\
\mathbf{v}
\end{bmatrix}
\hspace{1cm} (7)$$

Từ $(7)$ ta đi sẽ tìm $\mathbf{x}$ và $\mathbf{y}$:

Ở dòng thứ hai của $(7)$:

$$ \mathbf{y} = \mathbf{D}^{-1}(\mathbf{v}-\mathbf{B}^\mathsf{T}\mathbf{x}) \hspace{1cm} (8)$$

Thế $(8)$ vào dòng thứ nhất của $(7)$:

$$\mathbf{A} \mathbf{x} + \mathbf{B}\mathbf{D}^{-1}(\mathbf{v}-\mathbf{B}^\mathsf{T}\mathbf{x}) = \mathbf{u} \Leftrightarrow  (\mathbf{A} - \mathbf{B}\mathbf{D}^{-1}\mathbf{B}^\mathsf{T})\mathbf{x} = \mathbf{u} -  \mathbf{B}\mathbf{D}^{-1}\mathbf{v} \hspace{1cm} (9)$$

Kết hợp giữa $(8)$ và $(9)$ ta có được:

$$\mathbf{x} = (\mathbf{A} - \mathbf{B}\mathbf{D}^{-1}\mathbf{B}^\mathsf{T})^{-1}\mathbf{u} - (\mathbf{A} - \mathbf{B}\mathbf{D}^{-1}\mathbf{B}^\mathsf{T})^{-1}\mathbf{B}\mathbf{D}^{-1}\mathbf{v} \hspace{1cm} (10)$$

$$\mathbf{y=D^{-1}B^\mathsf{T}(A-BD^{-1}B^\mathsf{T})^{-1}u + (D^{-1} + 
D^{-1}B^\mathsf{T}(A-BD^{-1}B^\mathsf{T})^{-1}BD^{-1})v} \hspace{1cm} (11)$$

Ta đặt $\mathbf{L = (A-BD^{-1}B^\mathsf{T})^{-1}}$ để biểu thức đơn giản hơn, ta viết lại biểu thức $(10)$ và $(11)$.

$$\mathbf{x = Lu - LBD^{-1}v}\hspace{1cm} (12)$$

$$\mathbf{y=D^{-1}B^{\mathsf{T}}Lu + (D^{-1} + 
D^{-1}B^\mathsf{T}LBD^{-1})v} \hspace{1cm} (13)$$

Dễ dàng thấy được:

$$
\begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix} = 
\begin{bmatrix}
\mathbf{L} & \mathbf{LBD^{-1}} \\
\mathbf{D^{-1}B^{\mathsf{T}}L} & \mathbf{D^{-1} + 
D^{-1}B^\mathsf{T}LBD^{-1}}
\end{bmatrix}
\begin{bmatrix}
\mathbf{u} \\
\mathbf{v}
\end{bmatrix}
\hspace{1cm} (14)$$

Kết hợp giữa $(5)$ và $(14)$, ta có được:

$$
\mathbf{M} = 
\begin{bmatrix}
\mathbf{A} & \mathbf{B} \\
\mathbf{B}^\mathsf{T} & \mathbf{D}
\end{bmatrix}^{-1} = 
\begin{bmatrix}
\mathbf{L} & \mathbf{LBD^{-1}} \\
\mathbf{D^{-1}B^{\mathsf{T}}L} & \mathbf{D^{-1} + 
D^{-1}B^\mathsf{T}LBD^{-1}}
\end{bmatrix}
\space với\space \mathbf{L = (A-BD^{-1}B^\mathsf{T})^{-1}} 
\hspace{1cm} (15)
$$

Thế là ta đã xong với việc biểu diễn một ma trận nghịch đảo của ma trận vuông $\mathbf{M}$ dựa vào 4 khối ma trận con bên trong nó. Bạn có thấy ma trận này quen không, đúng rồi đấy nó chính là ma trận covariance.

Với 
$\mathbf{z} = \begin{bmatrix}
\mathbf{x} \\
\mathbf{y}
\end{bmatrix};\space \mathbf{x}\in \mathbb{R}^{p}, \space \mathbf{y}\in \mathbb{R}^{q}$ và 
$\mathbf{z} \thicksim \mathcal{N}(\mu_z,\Sigma_z)
\Leftrightarrow
\mathbf{z} \thicksim \mathcal{N}(
\begin{bmatrix}
\mu_x \\
\mu_y
\end{bmatrix}, 
\begin{bmatrix}
\Sigma_{xx} & \Sigma_{xy} \\
\Sigma_{yx} & \Sigma_{yy}
\end{bmatrix})$

## 2.2 Chứng minh 






