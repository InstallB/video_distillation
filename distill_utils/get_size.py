from PIL import Image
from torchvision import transforms

# Step 1: 定义 ImageNet Transform
imagenet_transform = transforms.Compose([
    
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 归一化
])

# Step 2: 读取一张 JPEG 图片
image_path = "/mnt/nas-new/home/yangnianzu/cyf/video_distillation/distill_utils/distill_rded_jpeg/ApplyEyeMakeup0/frame000001.jpg"  # 图片路径
image = Image.open(image_path).convert("RGB")  # 确保转换为 RGB 模式

# Step 3: 应用 ImageNet Transform
transformed_image = imagenet_transform(image)  # 图片变换

# Step 4: 获取图片的大小
# Transformed 图片是一个张量，维度为 (C, H, W)
print("Transformed Image Size:", transformed_image.size())
