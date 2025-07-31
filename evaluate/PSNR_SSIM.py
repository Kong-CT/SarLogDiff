import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import greycomatrix, greycoprops
import glob

def preprocess_image(image_path, size=(256, 256)):
    # 读取图像并转换为灰度
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 调整图像大小
    img_resized = cv2.resize(img, size)
    return img_resized

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value

def calculate_ssim(img1, img2):
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

def calculate_texture_similarity(img1, img2, distances=[5], angles=[0], levels=256):
   
    glcm1 = greycomatrix(img1, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    glcm2 = greycomatrix(img2, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    
    texture_features = ['contrast', 'homogeneity', 'energy', 'correlation']
    similarities = {}
    
    for feature in texture_features:
        prop1 = greycoprops(glcm1, feature).mean()
        prop2 = greycoprops(glcm2, feature).mean()
        
        similarities[feature] = 1 - abs(prop1 - prop2)
    
    return similarities

def compare_image_sets(real_image_paths, generated_image_paths):
    psnr_values = []
    ssim_values = []
    texture_similarities = {'contrast': [], 'homogeneity': [], 'energy': [], 'correlation': []}
    
    for real_img_path, gen_img_path in zip(real_image_paths, generated_image_paths):
        real_img = preprocess_image(real_img_path)
        gen_img = preprocess_image(gen_img_path)
        
        # 计算 PSNR 和 SSIM
        psnr_values.append(calculate_psnr(real_img, gen_img))
        ssim_values.append(calculate_ssim(real_img, gen_img))
        
        # 计算纹理相似性
        texture_similarity = calculate_texture_similarity(real_img, gen_img)
        for key in texture_similarity:
            texture_similarities[key].append(texture_similarity[key])
    
    # 计算平均 PSNR、SSIM 和纹理相似性
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_texture_similarities = {key: np.mean(values) for key, values in texture_similarities.items()}
    
    return avg_psnr, avg_ssim, avg_texture_similarities


real_image_paths = glob.glob('data/real_images/*.tif')

generated_image_paths = glob.glob('data/generated_images/*.tif')  
real_image_paths.sort()
generated_image_paths.sort()
num_images = min(len(real_image_paths), len(generated_image_paths))
real_image_paths = real_image_paths[:num_images]
generated_image_paths = generated_image_paths[:num_images]


avg_psnr, avg_ssim, avg_texture_similarities = compare_image_sets(real_image_paths, generated_image_paths)

# 输出结果
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print("Texture Similarities:")
for key, value in avg_texture_similarities.items():
    print(f"{key}: {value:.4f}")
