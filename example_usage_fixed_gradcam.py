
"""
使用修复后的 PoundNet GradCAM 的简单示例
Example usage of the fixed PoundNet GradCAM implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.gradcam.poundnet_gradcam_fixed import PoundNetGradCAM
from networks.poundnet_detector import PoundNet
from omegaconf import OmegaConf

def load_and_preprocess_image(image_path):
    """加载并预处理图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    
    # 转换为tensor并标准化 (CLIP标准化)
    image_array = np.array(image) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image_normalized = (image_array - mean) / std
    
    # 转换为PyTorch tensor
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float().unsqueeze(0)
    
    return image_tensor, np.array(image)

def visualize_cam_overlay(original_image, cam, alpha=0.4):
    """将CAM叠加到原始图像上"""
    # 标准化CAM到0-1范围
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # 创建热力图
    import matplotlib.cm as cm
    heatmap = cm.jet(cam_normalized)[:, :, :3]  # 移除alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加到原始图像
    overlay = original_image * (1 - alpha) + heatmap * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay, heatmap

def main():
    print("=== 使用修复后的 PoundNet GradCAM ===")
    
    # 1. 加载模型
    print("1. 加载 PoundNet 模型...")
    cfg = OmegaConf.load('./cfgs/poundnet.yaml')
    model = PoundNet(cfg)
    model.eval()
    
    # 2. 创建修复后的 GradCAM 实例
    print("2. 创建修复后的 GradCAM 实例...")
    gradcam = PoundNetGradCAM(model)
    
    # 3. 创建测试图像（如果没有真实图像的话）
    print("3. 创建测试图像...")
    test_image = torch.randn(1, 3, 224, 224)
    original_image = (torch.randn(224, 224, 3) * 0.5 + 0.5).clamp(0, 1)
    original_image = (original_image * 255).numpy().astype(np.uint8)
    
    # 如果你有真实图像，可以使用以下代码：
    # image_path = "path/to/your/image.jpg"
    # test_image, original_image = load_and_preprocess_image(image_path)
    
    # 4. 生成预测和CAM
    print("4. 生成预测和CAM...")
    
    # 获取模型预测
    with torch.no_grad():
        output = model(test_image)
        logits = output['logits'] if isinstance(output, dict) else output
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = logits.argmax(dim=1).item()
    
    print(f"预测结果: {'Real' if predicted_class == 0 else 'Fake'}")
    print(f"Real 概率: {probabilities[0]:.4f}")
    print(f"Fake 概率: {probabilities[1]:.4f}")
    
    # 5. 生成不同类别的CAM
    print("5. 生成 GradCAM 可视化...")
    
    # Real类别的CAM
    real_cam = gradcam.generate_cam(test_image, target_class='real')
    print(f"Real CAM 统计: min={real_cam.min():.4f}, max={real_cam.max():.4f}, mean={real_cam.mean():.4f}")
    
    # Fake类别的CAM
    fake_cam = gradcam.generate_cam(test_image, target_class='fake')
    print(f"Fake CAM 统计: min={fake_cam.min():.4f}, max={fake_cam.max():.4f}, mean={fake_cam.mean():.4f}")
    
    # 6. 可视化结果
    print("6. 创建可视化图像...")
    
    # 创建叠加图像
    real_overlay, real_heatmap = visualize_cam_overlay(original_image, real_cam)
    fake_overlay, fake_heatmap = visualize_cam_overlay(original_image, fake_cam)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：Real类别
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(real_heatmap)
    axes[0, 1].set_title(f'Real CAM\n(概率: {probabilities[0]:.3f})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(real_overlay)
    axes[0, 2].set_title('Real CAM 叠加')
    axes[0, 2].axis('off')
    
    # 第二行：Fake类别
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('原始图像')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fake_heatmap)
    axes[1, 1].set_title(f'Fake CAM\n(概率: {probabilities[1]:.3f})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(fake_overlay)
    axes[1, 2].set_title('Fake CAM 叠加')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'修复后的 PoundNet GradCAM 可视化\n预测: {["Real", "Fake"][predicted_class]} (置信度: {probabilities[predicted_class]:.3f})',
                 fontsize=16)
    plt.tight_layout()
    
    # 保存结果
    output_path = 'fixed_gradcam_result.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果已保存到: {output_path}")
    
    # 显示图像（如果在交互环境中）
    try:
        plt.show()
    except:
        print("无法显示图像，但已保存到文件")
    
    # 7. 生成对比分析
    print("7. 生成对比分析...")
    comparative_cams = gradcam.generate_comparative_cam(test_image)
    
    print("对比分析完成:")
    print(f"  Real CAM 范围: [{comparative_cams['real'].min():.4f}, {comparative_cams['real'].max():.4f}]")
    print(f"  Fake CAM 范围: [{comparative_cams['fake'].min():.4f}, {comparative_cams['fake'].max():.4f}]")
    
    # 8. 使用预测功能
    print("8. 使用集成预测功能...")
    prediction_result = gradcam.generate_prediction_with_cam(test_image)
    
    print("集成预测结果:")
    print(f"  预测类别: {prediction_result['predicted_class_name']}")
    print(f"  置信度: {prediction_result['confidence']:.4f}")
    print(f"  是否高置信度: {prediction_result['is_confident']}")
    
    print("\n✅ 修复后的 GradCAM 使用示例完成！")
    print("现在你可以使用 PoundNetGradCAM 生成有意义的可视化结果了。")

if __name__ == "__main__":
    main()