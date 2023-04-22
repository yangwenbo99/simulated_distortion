''' distortions,py methods for generating simulated distortions

Acknowledgement:  We used Cursor.so to write part of the code. 
'''
import torch 
import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import cv2

def degradation_simulated(
        img: np.ndarray, 
        save_to: str, 
        focus_blur_prob=0.7, 
        motion_blur_prob=0.5, 
        exposure_prob=0.2, 
        gaussian_noise_prob=0.4, 
        sensor_gaussian_exponential_beta=0.03,
        flicker_noise_prob=0.3, 
        override_jpeg_quality=-1,
        sf=2): 
    '''
    @param img: float type image
    @param sf: the ratio of downsampling and upsampling (not implemented yet)
    '''
    # Make sure img has BGR colour
    if img.shape[-1] == 1 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 3 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


    ori = img
    parameters = {} 

    # 1. Add focus blur, 
    #    currently we use Gaussian blur to similate.  
    #    This is the default method GIMP used to add focus bluyr
    if random.random() > focus_blur_prob: 
        parameters['focus'] = None
    else: 
        kernel_size = np.random.choice([21, 31, 41])
        radius = random.uniform(0, 30) 
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), radius)
        parameters['focus'] = {
                'method': 'gaussian', 
                'kernel_size': int(kernel_size), 
                'radius': float(radius)
                }

    # 2. Add motion blur
    if random.random() > motion_blur_prob: 
        parameters['motion'] = None
    else: 
        angle = np.random.randint(0, 360)
        kernel_size = np.random.randint(5, 40)
        motion_blur_kernel = np.zeros((kernel_size, kernel_size))
        motion_blur_kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, cv2.getRotationMatrix2D((int((kernel_size-1)/2), int((kernel_size-1)/2)), angle, 1.0), (kernel_size, kernel_size))
        motion_blur_kernel = motion_blur_kernel / kernel_size
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        parameters['motion'] = {
                'method': 'simple', 
                'angle': int(angle), 
                'kernel_size': int(kernel_size)
                }

    # 3. Adding over/under-exposure
    if random.random() > exposure_prob: 
        parameters['exposure'] = None
    else: 
        alpha = np.random.uniform(0.8, 1.2)  # scale 
        margin = (1 - alpha) / 2 
        beta = margin + margin * random.uniform(-1, 1) # offset 
        img = img * alpha + beta
        parameters['exposure'] = {
                'alpha': float(alpha), 
                'beta': float(beta)
                }

    # 4. Add sensor noise
    # See https://isl.stanford.edu/~abbas/ee392b/lect06.pdf

    h, w, c = img.shape
    # 4.1 Add 1/f noise      (flicker noise)
    # No much effects at this moment, maybe wrong parameters? 
    if random.random() > flicker_noise_prob:
        flicker_strength = 0
    else:
        flicker_strength = random.uniform(0, 1)
        x = np.linspace(-w/2, w/2, w)
        y = np.linspace(-h/2, h/2, h)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)
        F = np.random.normal(0, 1, img.shape)
        s_f = 1 / R 
        s_f = np.nan_to_num(s_f)
        # s_f /= s_f.max()
        F = F * s_f.reshape((h, w, -1)) * flicker_strength
        F = np.fft.ifftshift(F)
        img = img + np.real(np.fft.ifft2(F))
    # 4.2 Use Gaussian noise to approximate thermal noise 
    gauss = np.random.normal(0, 1, img.shape)
    if random.random() > gaussian_noise_prob: 
        gauss_strength = 0
    else: 
        gauss_strength = random.expovariate(1 / sensor_gaussian_exponential_beta)
    img = img + gauss * gauss_strength

    parameters['noise'] = {
            'flicker_strength': flicker_strength, 
            'gaussian_strength': gauss_strength, 
            }

    # 5. add JPEG distortion
    quality = np.random.randint(30, 97)
    if override_jpeg_quality > 0: 
        quality = override_jpeg_quality
    parameters['jpeg'] = {
            'quality': int(quality)
            }

    img = img.clip(0, 1)
    img = (img * 255).astype(np.uint8)

    if save_to: 
        parameters['saved_to'] = save_to
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(save_to, img, encode_param)
        return parameters
    else: 
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
        return ori, img, parameters

def degradation_simulated_deterministic(
        img: np.ndarray, parameters: dict, save_to: str = None):
    '''
    @param img: float type image
    @param parameters: a dictionary containing the degradation parameters
    @param sf: the ratio of downsampling and upsampling (not implemented yet)
    '''
    # Make sure img has BGR colour
    if img.shape[-1] == 1 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 3 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    ori = img

    # 1. Add focus blur
    if parameters['focus'] is None:
        pass
    else:
        kernel_size = parameters['focus']['kernel_size']
        radius = parameters['focus']['radius']
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), radius)

    # 2. Add motion blur
    if parameters['motion'] is None:
        pass
    else:
        angle = parameters['motion']['angle']
        kernel_size = parameters['motion']['kernel_size']
        motion_blur_kernel = np.zeros((kernel_size, kernel_size))
        motion_blur_kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, cv2.getRotationMatrix2D((int((kernel_size-1)/2), int((kernel_size-1)/2)), angle, 1.0), (kernel_size, kernel_size))
        motion_blur_kernel = motion_blur_kernel / kernel_size
        img = cv2.filter2D(img, -1, motion_blur_kernel)

    # 3. Adding over/under-exposure
    if parameters['exposure'] is None:
        pass
    else:
        alpha = parameters['exposure']['alpha']
        beta = parameters['exposure']['beta']
        img = img * alpha + beta

    # 4. Add sensor noise
    h, w, c = img.shape
    # 4.1 Add 1/f noise (flicker noise)
    flicker_strength = parameters['noise']['flicker_strength']
    if flicker_strength > 0:
        x = np.linspace(-w/2, w/2, w)
        y = np.linspace(-h/2, h/2, h)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)
        F = np.random.normal(0, 1, img.shape)
        s_f = 1 / R
        s_f = np.nan_to_num(s_f)
        F = F * s_f.reshape((h, w, -1)) * flicker_strength
        F = np.fft.ifftshift(F)
        img = img + np.real(np.fft.ifft2(F))
    # 4.2 Use Gaussian noise to approximate thermal noise
    gauss = np.random.normal(0, 1, img.shape)
    gauss_strength = parameters['noise']['gaussian_strength']
    img = img + gauss * gauss_strength

    # 5. add JPEG distortion
    quality = parameters['jpeg']['quality']

    img = img.clip(0, 1)
    img = (img * 255).astype(np.uint8)

    if save_to:
        parameters['saved_to'] = save_to
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(save_to, img, encode_param)
        return parameters
    else:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
        return ori, img, parameters

def denormalize_parameters(normalized_parameters: dict): 
    '''
    @param normalized_parameters: the normalized parameters. 
    Each numeric field is a float between 0 and 1.  With 0 be no distortion 
    and 1 be the most significant distortion. 
    # The structure of normalized_parameters is 
    normalized_parameters = {
        'focus': {
            'method': 'gaussian',  # the kernel size is always 41
            'radius': float  # between 0 and 1
            # 0 -- blur not applied
            # 1 -- radius = 30
        },
        'motion': {
            'method': 'simple',
            'angle': float,  # between 0 and 1
            'kernel_size': float  # between 0 and 1
            # 0 -- blur not applied
            # 1 -- radius = 30
        },
        'exposure': {
            'alpha': float,  # between 0 and 1
            'beta': float  # between 0 and 1
        },
        'noise': {
            'flicker_strength': 0,  # not implemented
            'gaussian_strength': float  # between 0 and 1
        },
        'jpeg': {
            'quality': float  # between 0 and 1
        }
    }

    Currently, flicker noise is not implemented. 
    '''
    pass

if __name__ == '__main__': 
    import os
    import json
    img = cv2.imread('test/example.jpg')
    img = img.astype(np.float32) / 255.0

    print(img.dtype)
    if not os.path.exists('test_out'):
        os.makedirs('test_out')
    ori_img, degraded_img, parameters = degradation_simulated(img, save_to='test_out/1_degraded_focus.jpg', focus_blur_prob=1)
    print(json.dumps(parameters, indent=4))
    ori_img, degraded_img, parameters = degradation_simulated(img, save_to='test_out/2_degraded_motion.jpg', motion_blur_prob=1)
    print(json.dumps(parameters, indent=4))
    ori_img, degraded_img, parameters = degradation_simulated(img, save_to='test_out/3_degraded_exposure.jpg', exposure_prob=1)
    print(json.dumps(parameters, indent=4))
    ori_img, degraded_img, parameters = degradation_simulated(
            img, save_to='test_out/4_degraded_flicker.jpg', 
            flicker_noise_prob=1, focus_blur_prob=0, motion_blur_prob=0, 
            gaussian_noise_prob=0, override_jpeg_quality=95)
    print(json.dumps(parameters, indent=4))
    ori_img, degraded_img, parameters = degradation_simulated(img, save_to='test_out/5_degraded_gaussian.jpg', gaussian_noise_prob=1)
    print(json.dumps(parameters, indent=4))
    ori_img, degraded_img, parameters = degradation_simulated(img, save_to='test_out/6_degraded_jpeg.jpg')
    print(json.dumps(parameters, indent=4))
