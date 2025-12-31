# GPA Prediction using Machine Learning

## 1. Giới thiệu đề tài
1.1. Bối cảnh và lý do chọn đề tài
Trong môi trường giáo dục đại học, kết quả học tập của sinh viên là một chỉ số quan trọng phản ánh mức độ tiếp thu kiến thức và hiệu quả rèn luyện trong quá trình học. Trong đó, điểm trung bình (GPA – Grade Point Average) thường được dùng như một thước đo tổng hợp để đánh giá năng lực học tập trong từng học kỳ hoặc toàn khóa.
Tuy nhiên, GPA không chỉ phụ thuộc vào năng lực học thuật mà còn chịu ảnh hưởng bởi nhiều yếu tố khác như: thói quen học tập, thời gian tự học, khả năng quản lý thời gian, mức độ căng thẳng, sức khỏe, lối sống, và mức độ hỗ trợ từ gia đình. Việc đánh giá các yếu tố này theo cách thủ công thường gặp khó khăn vì dữ liệu nhiều chiều, mối quan hệ phức tạp và có thể phi tuyến.
Vì vậy, việc áp dụng Trí tuệ nhân tạo (Artificial Intelligence) và Học máy (Machine Learning) để xây dựng mô hình dự đoán GPA là hướng tiếp cận phù hợp, giúp:
-Phân tích dữ liệu học tập một cách có hệ thống.
-Tìm ra các yếu tố có ảnh hưởng mạnh đến GPA.
-Dự đoán GPA của sinh viên dựa trên đặc trưng đầu vào, hỗ trợ cảnh báo sớm nguy cơ học lực giảm sút.
Đề tài “Dự đoán điểm trung bình (GPA Prediction) của sinh viên bằng các thuật toán Machine Learning” được lựa chọn nhằm minh họa quy trình xây dựng một hệ thống dự đoán theo hướng dữ liệu (data-driven), đồng thời đáp ứng yêu cầu thực hành các kỹ thuật AI/ML trong môn học.

1.2. Bài toán đặt ra
Bài toán của đề tài là bài toán hồi quy (Regression) trong học máy.
Cụ thể: từ các đặc trưng mô tả sinh viên (ví dụ: giờ học mỗi ngày, GPA kỳ trước, thời gian dùng mạng xã hội, mức độ áp lực học tập, tần suất tập thể dục, mức hỗ trợ từ gia đình, …), cần xây dựng mô hình dự đoán giá trị GPA (một biến số liên tục).
Đầu vào (Input X): tập các đặc trưng về học tập và lối sống.
Đầu ra (Output y): GPA dự đoán.

1.3. Ý nghĩa thực tiễn
Mặc dù là bài tập lớn trong phạm vi môn học, đề tài có ý nghĩa thực tiễn rõ ràng:
Đối với sinh viên: có thể tự đánh giá tình trạng học tập, nhận cảnh báo sớm khi mô hình dự đoán GPA giảm, từ đó điều chỉnh thói quen học tập/lối sống.
Đối với nhà trường/giảng viên: hỗ trợ theo dõi và phát hiện sớm nhóm sinh viên có nguy cơ học lực kém, giúp đưa ra kế hoạch hỗ trợ học tập phù hợp.
Đối với hướng nghiên cứu/ứng dụng AI: minh họa cách AI xử lý dữ liệu đa yếu tố, học mối quan hệ tuyến tính và phi tuyến, cũng như đánh giá mô hình bằng các chỉ số hồi quy.

1.4. Mục tiêu tổng quát
Xây dựng và đánh giá hệ thống dự đoán GPA của sinh viên dựa trên dữ liệu thói quen học tập và lối sống, sử dụng nhiều thuật toán Machine Learning nhằm so sánh hiệu quả và lựa chọn mô hình phù hợp.

1.5. Mục tiêu cụ thể
-Thu thập và mô tả dữ liệu phục vụ bài toán dự đoán GPA.
-Tiền xử lý dữ liệu (làm sạch, mã hóa biến phân loại, chuẩn hóa biến số).
-Phân tích và khám phá dữ liệu (EDA) để hiểu phân phối và mối quan hệ giữa các biến.
-Xây dựng và huấn luyện các mô hình hồi quy: Linear Regression, Random Forest, Gradient Boosting, XGBoost.
-Đánh giá và so sánh mô hình bằng các chỉ số: MAE, MSE, RMSE, R².
-Triển khai demo dự đoán để minh họa khả năng áp dụng thực tế
## 2. Dataset
Nguồn dữ liệu
Dữ liệu sử dụng trong đề tài được lấy từ bộ dữ liệu công khai trên Kaggle:
Student Habits and Academic Performance Dataset
Nguồn: Kaggle
Link tải:
👉 https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset
Bộ dữ liệu này cung cấp thông tin liên quan đến thói quen học tập, lối sống và kết quả học tập của sinh viên, phù hợp với bài toán dự đoán điểm trung bình (GPA Prediction).
Mô tả các thuộc tính dữ liệu
Dataset bao gồm các thuộc tính mô tả thông tin cá nhân, thói quen học tập, lối sống, sức khỏe tinh thần và kết quả học tập của sinh viên. Các cột dữ liệu chính được sử dụng trong đề tài như sau:
| Tên cột                         | Kiểu dữ liệu | Mô tả                                             |
| ------------------------------- | ------------ | ------------------------------------------------- |
| `study_hours_per_day`           | Numerical    | Số giờ học trung bình mỗi ngày                    |
| `social_media_hours`            | Numerical    | Thời gian sử dụng mạng xã hội mỗi ngày            |
| `netflix_hours`                 | Numerical    | Thời gian xem phim/giải trí mỗi ngày              |
| `part_time_job`                 | Categorical  | Trạng thái làm thêm (có/không)                    |
| `attendance_percentage`         | Numerical    | Tỷ lệ chuyên cần (%)                              |
| `diet_quality`                  | Categorical  | Chất lượng chế độ ăn uống                         |
| `exercise_frequency`            | Categorical  | Tần suất tập thể dục                              |
| `mental_health_rating`          | Numerical    | Mức độ sức khỏe tinh thần                         |
| `extracurricular_participation` | Categorical  | Mức độ tham gia hoạt động ngoại khóa              |
| `previous_gpa`                  | Numerical    | GPA của học kỳ trước                              |
| `stress_level`                  | Categorical  | Mức độ căng thẳng trong học tập                   |
| `dropout_risk`                  | Categorical  | Nguy cơ bỏ học                                    |
| `study_environment`             | Categorical  | Môi trường học tập                                |
| `access_to_tutoring`            | Categorical  | Khả năng tiếp cận gia sư/hỗ trợ học tập           |
| `parental_support_level`        | Categorical  | Mức độ hỗ trợ từ gia đình                         |
| `motivation_level`              | Categorical  | Mức độ động lực học tập                           |
| `exam_anxiety_score`            | Numerical    | Mức độ lo âu khi thi                              |
| `learning_style`                | Categorical  | Phong cách học tập                                |
| `time_management_score`         | Numerical    | Khả năng quản lý thời gian                        |
| `exam_score`                    | Numerical    | **Điểm trung bình (GPA) – biến mục tiêu dự đoán** |
Dataset bao gồm cả: Biến số (Numerical features), Biến phân loại (Categorical features) do đó cần thực hiện mã hóa và chuẩn hóa trong bước tiền xử lý.
## 3. Pipeline (Tiền xử lý → Huấn luyện → Đánh giá → Suy luận)
Pipeline của hệ thống được xây dựng theo quy trình Machine Learning chuẩn, đảm bảo tính nhất quán giữa dữ liệu huấn luyện và dữ liệu suy luận (inference). Toàn bộ pipeline bao gồm bốn giai đoạn chính: tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá mô hình và suy luận dự đoán.
Tiền xử lý dữ liệu (Data Preprocessing)

Ở bước tiền xử lý, dữ liệu đầu vào được chuẩn hóa và mã hóa để phù hợp với các thuật toán Machine Learning.
Các bước chính:
Phân tách biến đầu vào và biến mục tiêu
Biến đầu vào (X): các đặc trưng học tập, lối sống và hành vi của sinh viên.
Biến mục tiêu (y): GPA.
Xác định loại đặc trưng
Biến số (Numerical features): số giờ học, GPA kỳ trước, điểm thi, thời gian sử dụng mạng xã hội, …
Biến phân loại (Categorical features): mức độ căng thẳng, mức độ hỗ trợ gia đình, môi trường học tập, …
Chuẩn hóa dữ liệu số
Các biến số được chuẩn hóa bằng StandardScaler nhằm đưa dữ liệu về cùng thang đo, giúp mô hình học ổn định và hội tụ tốt hơn.
Mã hóa biến phân loại
Các biến phân loại được mã hóa bằng OneHotEncoder, giúp chuyển đổi dữ liệu dạng chuỗi sang dạng số để mô hình có thể xử lý.
Toàn bộ quá trình tiền xử lý được đóng gói trong một pipeline nhằm đảm bảo cùng một quy trình được áp dụng cho cả dữ liệu huấn luyện và dữ liệu mới.

3.2. Huấn luyện mô hình (Model Training)
Sau khi dữ liệu được tiền xử lý, tập dữ liệu được chia thành:
Tập huấn luyện (Training set)
Tập kiểm tra (Test set)
Trên tập huấn luyện, hệ thống tiến hành huấn luyện nhiều mô hình hồi quy khác nhau, bao gồm:
Linar Regression
Random Forest Regression
Gradient Boosting Regression
XGBoost Regression
Việc sử dụng nhiều mô hình cho phép so sánh khả năng học các mối quan hệ tuyến tính và phi tuyến giữa đặc trưng đầu vào và GPA.

3.3. Đánh giá mô hình (Model Evaluation)
Các mô hình sau khi huấn luyện được đánh giá trên tập kiểm tra, nhằm đảm bảo tính khách quan và khả năng tổng quát hóa.
Các chỉ số đánh giá được sử dụng:
MAE (Mean Absolute Error): đo sai số tuyệt đối trung bình.
MSE (Mean Squared Error): đo sai số bình phương trung bình.
RMSE (Root Mean Squared Error): căn bậc hai của MSE, giúp dễ diễn giải sai số.
R² (Coefficient of Determination): đo mức độ giải thích biến thiên của GPA.
Kết quả cho thấy các mô hình ensemble (Random Forest, Gradient Boosting, XGBoost) có khả năng dự đoán tốt hơn so với mô hình tuyến tính trong bài toán này.

3.4. Suy luận và triển khai (Inference & Deployment)
Sau khi lựa chọn mô hình phù hợp, hệ thống hỗ trợ dự đoán GPA cho dữ liệu mới thông qua  hình thức:
Ứng dụng Web Flask: Mô hình được tích hợp vào một ứng dụng web đơn giản, cho phép người dùng nhập thông tin sinh viên thông qua giao diện HTML và nhận kết quả dự đoán GPA trên trình duyệt.
Việc triển khai demo giúp minh họa khả năng ứng dụng thực tế của mô hình và hoàn thiện quy trình Machine Learning từ dữ liệu đến người dùng cuối.

## 4.Mô hình sử dụng và lí do chọn
Dựa trên kết quả phân tích dữ liệu, đặc điểm của bài toán dự đoán GPA (bài toán hồi quy) và mục tiêu so sánh hiệu quả giữa các thuật toán Machine Learning, đề tài lựa chọn và triển khai các mô hình sau:

4.1. Linear Regression
Linear Regression là mô hình hồi quy tuyến tính, giả định mối quan hệ tuyến tính giữa các đặc trưng đầu vào và biến mục tiêu GPA.
Lý do lựa chọn:
Là mô hình đơn giản, dễ triển khai và dễ diễn giải.
Được sử dụng làm mô hình cơ sở (baseline) để so sánh với các mô hình phức tạp hơn.
Giúp đánh giá mức độ tuyến tính trong mối quan hệ giữa các yếu tố học tập và GPA.
Vai trò trong đề tài:
Cung cấp mốc tham chiếu ban đầu để đánh giá mức độ cải thiện khi sử dụng các mô hình phi tuyến.

4.2. Random Forest Regression
Random Forest Regression là mô hình ensemble kết hợp nhiều cây quyết định thông qua kỹ thuật bagging, trong đó mỗi cây được huấn luyện trên một tập con ngẫu nhiên của dữ liệu và đặc trưng.
Lý do lựa chọn:
Có khả năng học các mối quan hệ phi tuyến và phức tạp giữa các đặc trưng.
Giảm hiện tượng overfitting so với cây quyết định đơn lẻ.
Hoạt động tốt với dữ liệu có nhiều đặc trưng và ít yêu cầu tiền xử lý phức tạp.
Vai trò trong đề tài:
Đánh giá hiệu quả của mô hình ensemble dựa trên bagging trong bài toán dự đoán GPA.

4.3. Gradient Boosting Regression
Gradient Boosting Regression là mô hình boosting, trong đó các cây quyết định được huấn luyện nối tiếp, mỗi cây mới tập trung học các sai số (residual) của mô hình trước đó.
Lý do lựa chọn:
Có khả năng cải thiện dần độ chính xác thông qua việc học từ sai số.
Phù hợp với các bài toán hồi quy có mối quan hệ phi tuyến.
Thường cho kết quả tốt hơn so với các mô hình đơn giản khi được tinh chỉnh tham số phù hợp.
Vai trò trong đề tài:
Đánh giá hiệu quả của phương pháp boosting trong việc nâng cao chất lượng dự đoán GPA.

4.4. XGBoost Regression
XGBoost (Extreme Gradient Boosting) là phiên bản nâng cao của Gradient Boosting, được tối ưu hóa về hiệu năng và khả năng tổng quát hóa.
Lý do lựa chọn:
Tích hợp cơ chế regularization (L1, L2) giúp giảm overfitting.
Sử dụng thuật toán tối ưu hóa hiệu quả, cho phép huấn luyện nhanh và chính xác.
Được sử dụng rộng rãi trong các bài toán học máy thực tế và đạt hiệu suất cao.
Vai trò trong đề tài:
Đóng vai trò là mô hình nâng cao để so sánh với các phương pháp khác và đánh giá mức hiệu quả cao nhất có thể đạt được trong bài toán dự đoán GPA.

4.5. Tổng kết lựa chọn mô hình
Việc sử dụng đồng thời nhiều mô hình từ đơn giản đến phức tạp mang lại các lợi ích sau:
So sánh trực quan giữa mô hình tuyến tính và phi tuyến.
Đánh giá tác động của các kỹ thuật ensemble và boosting.
Lựa chọn mô hình phù hợp nhất dựa trên các chỉ số đánh giá (MAE, RMSE, R²).
Qua đó, đề tài không chỉ tập trung vào độ chính xác mà còn hướng tới việc hiểu rõ ưu – nhược điểm của từng thuật toán trong bối cảnh bài toán dự đoán GPA.

## 5. Kết quả của các Metric đánh giá
| Mô hình                      | MAE       | MSE        | RMSE      | R²        |
| ---------------------------- | --------- | ---------- | --------- | --------- |
| Linear Regression            | 3.196     | 17.539     | 4.188     | 0.870     |
| KNN Regression               | 4.315     | 30.731     | 5.544     | 0.773     |
| Decision Tree Regression     | 3.442     | 20.344     | 4.510     | 0.850     |
| Random Forest Regression     | 3.239     | 17.489     | 4.182     | 0.871     |
| Gradient Boosting Regression | **3.224** | **17.320** | **4.162** | **0.872** |
| XGBoost Regression           | 3.241     | 17.501     | 4.183     | 0.871     |

## 6. Hướng dẫn chạy dự án
Cài đặt môi trường
1.1. Yêu cầu hệ thống
Python >= 3.8
pip

1.2. Cài đặt thư viện cần thiết
Clone repository về máy:
git clone https://github.com/huyhoang9965/BTLMachineLearning.git
cd BTLMachineLearning
Cài đặt các thư viện:
pip install -r requirements.txt

2. Huấn luyện mô hình (Training)
Toàn bộ quá trình huấn luyện và đánh giá mô hình được thực hiện trong thư mục app/.
Chạy notebook huấn luyện:
cd app
jupyter notebook
Mở file:
BTL AI.ipynb

Notebook này thực hiện các bước:
Tiền xử lý dữ liệu
Huấn luyện các mô hình hồi quy (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
Đánh giá mô hình bằng MAE, MSE, RMSE, R²
Lưu mô hình đã huấn luyện dưới dạng .pkl

3. Chạy demo / suy luận (Inference)

3.1. Chạy demo bằng script Python
Chạy file demo:
cd demo
python demo.py
File demo cho phép:
Nạp mô hình đã huấn luyện
Nhập dữ liệu mẫu
Xuất ra kết quả dự đoán GPA

3.2. Chạy demo bằng ứng dụng Web Flask
Từ thư mục gốc của dự án:
python demo.py
Mở trình duyệt và truy cập:
http://127.0.0.1:5000
Người dùng có thể nhập thông tin sinh viên thông qua giao diện HTML và nhận kết quả dự đoán GPA trực tiếp trên trình duyệt.

4. Ghi chú
Do giới hạn dung lượng của GitHub, các mô hình có kích thước lớn (ví dụ Random Forest) có thể không được đẩy trực tiếp lên repository.
Trong trường hợp đó, người dùng cần huấn luyện lại mô hình bằng notebook trước khi chạy demo.
Pipeline tiền xử lý và mô hình được đảm bảo thống nhất giữa quá trình huấn luyện và suy luận.

## 7. Cấu trúc thư mục dự án
BTLMachineLearning/
│
├── app/
│   (Thư mục dự kiến dùng để triển khai các bước tiền xử lý dữ liệu, huấn luyện dữ liệu
│     và mở rộng trong tương lai)
│
├── demo/
│   ├── BTLAI.ipynb
│   │   Notebook dùng để tiền xử lý dữ liệu, trực quan hóa, demo huấn luyện, đánh giá
│   │   và thử nghiệm dự đoán GPA
│   │
│   └── demo.py
│       Script Python dùng để chạy demo dự đoán GPA
│
├── templates/
│   └── index.html
│       Giao diện HTML cho ứng dụng demo dự đoán GPA
│
├── data/
│   └── sample_data.csv
│       Dữ liệu mẫu dùng để minh họa và kiểm thử
│       (dữ liệu đầy đủ được tải từ Kaggle)
│
├── reports/
│   └── BTLML.docx
│       Báo cáo bài tập lớn môn Trí tuệ nhân tạo
│
├── slides/
│   └── *.pdf
│       Slide thuyết trình bài tập lớn
│
├── README.md
│   Tài liệu mô tả đề tài, pipeline và hướng dẫn chạy dự án
│
├── requirements.txt
│   Danh sách các thư viện Python cần thiết
│
└── .gitignore
    Các file/thư mục không đẩy lên GitHub

## 8.Tác giả

Họ và tên: Vũ Huy Hoàng
Mã Lớp: 124231
Mã sinh viên: 12423073

