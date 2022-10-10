---
layout: post
author: nguyenthanhminh
title: Algorithm 1 - Kalman Filter Part 1
---

# 1. Đôi nét về Kalman Filter

Bạn có biết rằng Kalman Filter được sử dụng để dẫn đường cho
chiến dịch Apolo vào năm 1960. Vậy bộ lọc này phải được chứng minh
là đủ manh mẽ để được áp dựng vào trong những chuyến hành trình
trong không gian như vậy.

## Đầu tiên mình sẽ tóm gọn lại Kalman Filter là gì:

Kalman Filter thực chất là một phương pháp dự đoán trạng thái tiếp theo của object, và dựa vào quan sát của hệ thống để có thê tinh chỉnh lại các thông số, tăng độ tin cây cho lần dự đoán tiếp theo. Ví dụ để dễ hiểu hơn: Khi bạn chơi bóng đá, ban sẽ dự đoán vị trí tiếp theo của quả bóng để đỡ đúng không nào ️⚽ (đây gọi là dự đoán), nhưng đột
nhiên banh lại đi lệch một chút so với dự đoán của bạn 💨⚽ (đây gọi là quan sát) vậy ở lần dự đoán tiếp theo bạn sẽ sử dụng thông tin của quả bóng bị lệch hướng để có được dự đoán chính xác cao. Dễ hiểu phải không nào. Vậy thì làm sao mà Kalman Filter làm được như thế, hãy cùng mình
đi sâu vào bài toán nhé 🤓🤓🤓.

Chúng ta bắt đầu với dự đoán torng Kalman Filter.
Giả sử ta có vector chứa thông tin về vị trí và vận tốc của quả bóng: 
$$ \mathbf{x} = \begin{bmatrix} 
position \\
velocity
\end{bmatrix} $$ 

hay tổng quát hơn $\mathbf{x} \in \mathbb{R}^{n}$ chứa n thông tin của $\mathbf{x}$ cần dự đoán. 

Ma trận $ \mathbf{F} \in \mathbb{R}^{n*n}$ dùng tổ hợp tuyến tính của $\mathbf{n}$ thông tin từ $\mathbf{x}$ để dự đoán tráng thái (state) tiếp theo. 

Ví dụ ta xem như quả bóng chuyển động đều, dễ thấy rằng: $position^{t+1} = position^{t} + velocity$ giả sử $t = 1$. Lúc này ma trận $ \mathcal{F} $ của chúng ta sẽ là: 
$$ \mathbf{F} = \begin{bmatrix}
1&1\\ 
0&1
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
\end{bmatrix} \\ = 
\begin{bmatrix}
position^{t} + velocity^{t}\\
velocity^{t}
\end{bmatrix}
= 
\begin{bmatrix}
position^{t + 1}\\
velocity
\end{bmatrix}
$$ 

Tiếp theo đến quan sát $\mathbf{y}$, ở đây ví dụ như thứ ta quan sát được là vị trị của quả bóng chẳng hạn. 
$$ \mathbf{y} = \begin{bmatrix}
position \\
\end{bmatrix} $$ 

Các bạn sẽ tự hỏi tại sao số chiều của quan sát (observer) và dự đoán (estimator) lại khác nhau. Vì mô hình muốn quan sát những thông tin mà các hệ thống không thể quan sát được và thực tế thông tin quan sát được ít hơn so với những gì muốn đo lường.