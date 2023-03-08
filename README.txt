Đồ án cuối kỳ môn CS336.N11.KHTN (Truy vấn thông tin đa phương tiện)

Project 2: IMAGE RETRIEVAL

Thành viên nhóm:
	- 20520208 - Lê Nhật Kha
	- 20520347 - Lê Xuân Tùng
	- 20520435 - Nguyễn Duy Đạt

Quy trình khởi chạy hệ thống:

1. Download The Paris Dataset vào thư mục .\dataset\paris
Link dataset: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/

2. Download groundtruth của The Paris Dataset vào thư mục .\dataset\groundtruth
Link: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/

3. Trích xuất đặc trưng dữ liệu ảnh, run command python indexing.py --feature_extractor Resnet50

4. Evaluate trên tập query, run command python ranking.py --feature_extractor Resnet50

5. Tính toán mAP, run command python evaluate.py --feature_extractor Resnet50

6. Chạy demo với giao diện streamlit, run command streamlit run demo.py