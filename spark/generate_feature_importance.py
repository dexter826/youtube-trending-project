"""
Generate Feature Importance Chart
Tải mô hình đã được huấn luyện, trích xuất thông tin về mức độ quan trọng của các đặc trưng,
và vẽ biểu đồ bằng matplotlib.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
from pyspark.ml import PipelineModel


class FeatureImportanceGenerator:
    def __init__(self):
        """Khởi tạo generator với Spark session"""
        self.spark = get_spark_session("FeatureImportanceGenerator", PRODUCTION_CONFIGS["ml_training"])
        self.hdfs_base_path = "hdfs://localhost:9000/youtube_trending"
        self.models_path = f"{self.hdfs_base_path}/models"
        self.output_dir = Path(__file__).parent
        
    def load_model(self):
        """Tải mô hình Random Forest Regression đã được huấn luyện từ HDFS"""
        try:
            model_path = f"{self.models_path}/days_regression"
            print(f"Đang tải mô hình từ: {model_path}")
            model = PipelineModel.load(model_path)
            print("✓ Mô hình đã được tải thành công!")
            return model
        except Exception as e:
            print(f"✗ Lỗi khi tải mô hình: {str(e)}")
            return None
    
    def extract_feature_importance(self, model):
        """Trích xuất thông tin feature importance từ Random Forest model"""
        try:
            # Lấy Random Forest stage từ pipeline (stage cuối cùng)
            rf_model = model.stages[-1]
            
            # Lấy feature importances
            feature_importances = rf_model.featureImportances.toArray()
            
            # Lấy VectorAssembler stage để biết tên các features
            assembler = model.stages[0]
            feature_names = assembler.getInputCols()
            
            print(f"\n✓ Đã trích xuất {len(feature_names)} features:")
            
            # Tạo dictionary mapping feature name -> importance
            importance_dict = {}
            for idx, name in enumerate(feature_names):
                importance_dict[name] = feature_importances[idx]
            
            # Sắp xếp theo importance giảm dần
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # In ra top 10 features quan trọng nhất
            print("\nTop 10 features quan trọng nhất:")
            for i, (name, importance) in enumerate(sorted_importance[:10], 1):
                print(f"{i:2d}. {name:30s}: {importance:.6f}")
            
            return sorted_importance
            
        except Exception as e:
            print(f"✗ Lỗi khi trích xuất feature importance: {str(e)}")
            return None
    
    def plot_feature_importance(self, feature_importance_list, top_n=5):
        """
        Vẽ biểu đồ feature importance
        
        Args:
            feature_importance_list: List of tuples (feature_name, importance_value)
            top_n: Số lượng features quan trọng nhất để hiển thị
        """
        try:
            # Lấy top N features
            top_features = feature_importance_list[:top_n]
            top_features_reversed = list(reversed(top_features))
            features = [item[0] for item in top_features_reversed]
            importances = [item[1] for item in top_features_reversed]
            
            # Tạo figure và axis
            plt.figure(figsize=(12, 6))
            
            # Tạo horizontal bar chart
            y_pos = np.arange(len(features))
            bars = plt.barh(y_pos, importances, alpha=0.8)
            
            # Tô màu gradient cho bars
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Cài đặt labels và title
            plt.yticks(y_pos, features)
            plt.xlabel('Mức độ quan trọng (Importance)', fontsize=12, fontweight='bold')
            plt.ylabel('Đặc trưng (Features)', fontsize=12, fontweight='bold')
            plt.title(f'Top {top_n} Đặc Trưng Quan Trọng Nhất\n(Random Forest Regression Model)', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # Thêm grid để dễ đọc
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Thêm giá trị importance vào cuối mỗi bar
            for i, (feature, importance) in enumerate(top_features_reversed):
                plt.text(importance, i, f' {importance:.4f}', 
                        va='center', fontsize=9, fontweight='bold')
            
            # Điều chỉnh layout để tránh bị cắt
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = self.output_dir / "feature_importance_chart.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Biểu đồ đã được lưu tại: {output_path}")
            
            # Đóng figure để giải phóng memory
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            print(f"✗ Lỗi khi vẽ biểu đồ: {str(e)}")
            return None
    
    def generate_chart(self, top_n=5):
        """
        Chạy toàn bộ quy trình: tải model, trích xuất importance, và vẽ biểu đồ
        
        Args:
            top_n: Số lượng features quan trọng nhất để hiển thị (mặc định: 5)
        """
        print("=" * 70)
        print("CHƯƠNG TRÌNH TẠO BIỂU ĐỒ FEATURE IMPORTANCE")
        print("=" * 70)
        
        # Bước 1: Tải mô hình
        print("\n[Bước 1/3] Đang tải mô hình...")
        model = self.load_model()
        if model is None:
            print("\n✗ Không thể tải mô hình. Vui lòng kiểm tra lại đường dẫn HDFS.")
            return False
        
        # Bước 2: Trích xuất feature importance
        print("\n[Bước 2/3] Đang trích xuất feature importance...")
        feature_importance = self.extract_feature_importance(model)
        if feature_importance is None:
            print("\n✗ Không thể trích xuất feature importance.")
            return False
        
        # Bước 3: Vẽ và lưu biểu đồ
        print(f"\n[Bước 3/3] Đang vẽ biểu đồ với top {top_n} features...")
        chart_path = self.plot_feature_importance(feature_importance, top_n=top_n)
        if chart_path is None:
            print("\n✗ Không thể tạo biểu đồ.")
            return False
        
        print("\n" + "=" * 70)
        print("✓ HOÀN THÀNH! Biểu đồ đã được tạo thành công.")
        print("=" * 70)
        
        # Đóng Spark session
        self.spark.stop()
        return True


def main():
    """Main function"""
    try:
        # Tạo generator instance
        generator = FeatureImportanceGenerator()
        
        # Chạy quy trình tạo biểu đồ (hiển thị top 5 features)
        success = generator.generate_chart(top_n=5)
        
        if success:
            print("\n✓ Script đã chạy thành công!")
            return 0
        else:
            print("\n✗ Script gặp lỗi trong quá trình thực thi.")
            return 1
            
    except Exception as e:
        print(f"\n✗ Lỗi không mong đợi: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
