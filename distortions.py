''' distortions,py methods for generating simulated distortions

Acknowledgement:  We used Cursor.so to write part of the code. 
'''
from typing import List
import torch 
import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import cv2
import kornia

def degradation_simulated(
        img: np.ndarray, 
        save_to: str, 
        focus_blur_prob=0.7, 
        motion_blur_prob=0.5, 
        exposure_prob=0.2, 
        gaussian_noise_prob=0.4, 
        flicker_noise_prob=0.3, 
        override_jpeg_quality=-1,
        sf=2,
        max_focus_blur_radius=30, 
        max_motion_blur_kernel_size=20, 
        max_exposure_scale_change=0.2, 
        sensor_gaussian_exponential_beta=0.03,
        min_jpeg_quality=30,
        max_jpeg_quality=97,
        ): 
    '''
    @param img: float type image
    @param sf: the ratio of downsampling and upsampling (not implemented yet)
    @param max_motion_blur_kernel_size: max size of (half) of the motion blur kernel
    @param sensor_gaussian_exponential_beta: mean of Gaussian noise strength.
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
        radius = random.uniform(0, max_focus_blur_radius) 
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
        kernel_size = np.random.randint(1, max_motion_blur_kernel_size)
        kernel_size = kernel_size * 2 + 1
        direction = random.uniform(-1, 1) 
        motion_blur = kornia.filters.MotionBlur(kernel_size, angle, direction, border_type='reflect')
        img_tensor = kornia.image_to_tensor(img, keepdim=False)
        img_tensor = motion_blur(img_tensor)
        img = kornia.tensor_to_image(img_tensor)
        parameters['motion'] = {
                'method': 'simple', 
                'angle': int(angle), 
                'kernel_size': int(kernel_size), 
                'direction': float(direction)
                }

    # 3. Adding over/under-exposure
    if random.random() > exposure_prob: 
        parameters['exposure'] = None
    else: 
        alpha = np.random.uniform(1 - max_exposure_scale_change,
                                  1 + max_exposure_scale_change)  # scale 
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
    quality = np.random.randint(min_jpeg_quality, max_jpeg_quality)
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
    import kornia

    if parameters['motion'] is None or parameters['motion']['kernel_size'] == 1:
        pass
    else:
        angle = parameters['motion']['angle']
        kernel_size = parameters['motion']['kernel_size']
        direction = parameters['motion']['direction']
        motion_blur = kornia.filters.MotionBlur(kernel_size, angle, direction, border_type='reflect')
        # With these lines using kornia.tensor_to_image and kornia.image_to_tensor:
        img_tensor = kornia.image_to_tensor(img, keepdim=False)
        img_tensor = motion_blur(img_tensor)
        img = kornia.tensor_to_image(img_tensor)
        # img = motion_blur(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()

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
            'angle': float,  # between 0 and 360, sent to target func iton as is. 
            'kernel_size': float,  # between 0 and 1
            # 0 -- blur not applied (passed as kernel_size=1)
            # 1 -- radius = 31
            'direction': float  # between -1 and 1, sent to target func iton as is. 
        },
        'exposure': {
            'alpha': float,  # between 0 and 1
            'alpha_type': 'over' or 'under', 
            # alpha = 0 will be converted to 1
            # If alpha_type type is over, alpha = 1 will be converted to 3
            # If alpha_type type is under, alpha = 1 will be converted to 1/3
            'beta': float  # between 0 and 1
            'beta_type': 'over' or 'under', 
            # beta = 0 will be converted to (1 - alpha) / 2, such that 
            # the shifted range is centered. 
            # If beta_type type is over, beta = 1 will be converted to 1
            # If beta_type type is under, beta = 1 will be converted to - alpha
        },
        'noise': {
            'flicker_strength': 0,  # not implemented
            'gaussian_strength': float  # between 0 and 1
            # gaussian_strength will be converted to F^(-1) (gaussian_strength),
            # where F^(-1) is the inverse CDF of exponential distribution
            # with lambda = 1
        },
        'jpeg': {
            'quality': float  # between 0 and 1
            # 0 -- quality == 97
            # 1 -- quality == 10
        }
    }

    Currently, flicker noise is not implemented. 
    '''
    denormalized_parameters = {}

    # Denormalize focus parameters
    if normalized_parameters['focus']['radius'] > 0:
        denormalized_parameters['focus'] = {
            'method': normalized_parameters['focus']['method'],
            'kernel_size': 41,
            'radius': normalized_parameters['focus']['radius'] * 30
        }
    else: 
        denormalized_parameters['focus'] = None

    # Denormalize motion parameters
    if normalized_parameters['motion']['kernel_size'] > 0: 
        denormalized_parameters['motion'] = {
            'method': normalized_parameters['motion']['method'],
            'angle': normalized_parameters['motion']['angle'], 
            'kernel_size': int(normalized_parameters['motion']['kernel_size'] * 30) // 2 * 2 + 1,
            'direction': normalized_parameters['motion']['direction']

        }
    else: 
        denormalized_parameters['motion'] = None

    # Denormalize exposure parameters
    if normalized_parameters['exposure'] is not None:
        alpha = normalized_parameters['exposure']['alpha']
        if normalized_parameters['exposure']['alpha_type'] == 'over':
            alpha = 1 + alpha * 2
        else:
            alpha = 1 / (1 + alpha * 2)

        beta = normalized_parameters['exposure']['beta']
        if normalized_parameters['exposure']['beta_type'] == 'over':
            beta = (1 - alpha) / 2 + beta * (1 - (1 - alpha) / 2)
        else:
            beta = (1 - alpha) / 2 - beta * (1 - (1 - alpha) / 2)

        denormalized_parameters['exposure'] = {
            'alpha': alpha,
            'beta': beta
        }
    else: 
        denormalized_parameters['exposure'] = None

    # Denormalize noise parameters
    denormalized_parameters['noise'] = {
        'flicker_strength': normalized_parameters['noise']['flicker_strength'],
        'gaussian_strength': -np.log(1 - normalized_parameters['noise']['gaussian_strength']) / 20
    }

    # Denormalize JPEG parameters
    denormalized_parameters['jpeg'] = {
        'quality': int(97 - normalized_parameters['jpeg']['quality'] * 87)
    }

    return denormalized_parameters

def normalize_parameters(denormalized_parameters: dict): 
    '''
    @param denormalized_parameters: the denormalized parameters.
    Each numeric field is a float or int with its original value.
    The structure of denormalized_parameters is the same as the output of denormalize_parameters() function.

    This function converts the denormalized parameters back to normalized parameters.
    Each numeric field will be a float between 0 and 1. With 0 be no distortion
    and 1 be the most significant distortion.

    Currently, flicker noise is not implemented.
    '''
    normalized_parameters = {}

    # Normalize focus parameters
    if denormalized_parameters['focus'] is not None:
        normalized_parameters['focus'] = {
            'method': denormalized_parameters['focus']['method'],
            'radius': denormalized_parameters['focus']['radius'] / 30
        }
    else:
        normalized_parameters['focus'] = {'radius': 0}

    # Normalize motion parameters
    if denormalized_parameters['motion'] is not None:
        normalized_parameters['motion'] = {
            'method': denormalized_parameters['motion']['method'],
            'angle': denormalized_parameters['motion']['angle'],
            'kernel_size': denormalized_parameters['motion']['kernel_size'] / 30,
            'direction': denormalized_parameters['motion']['direction']
        }
    else:
        normalized_parameters['motion'] = {'kernel_size': 0}

    # Normalize exposure parameters
    if denormalized_parameters['exposure'] is not None:
        alpha = denormalized_parameters['exposure']['alpha']
        if alpha >= 1:
            alpha_type = 'over'
            alpha = (alpha - 1) / 2
        else:
            alpha_type = 'under'
            alpha = 1 - 2 * alpha

        beta = denormalized_parameters['exposure']['beta']
        if beta >= (1 - alpha) / 2:
            beta_type = 'over'
            beta = (beta - (1 - alpha) / 2) / (1 - (1 - alpha) / 2)
        else:
            beta_type = 'under'
            beta = 1 - 2 * beta / (1 - alpha)

        normalized_parameters['exposure'] = {
            'alpha': alpha,
            'alpha_type': alpha_type,
            'beta': beta,
            'beta_type': beta_type
        }

    # Normalize noise parameters
    normalized_parameters['noise'] = {
        'flicker_strength': denormalized_parameters['noise']['flicker_strength'],
        'gaussian_strength': 1 - np.exp(-denormalized_parameters['noise']['gaussian_strength'] * 0.03)
    }

    # Normalize JPEG parameters
    normalized_parameters['jpeg'] = {
        'quality': (97 - denormalized_parameters['jpeg']['quality']) / 87
    }

    return normalized_parameters


def random_parameter_group(distortions: List[str], n=8, ordered=True, 
        focus_blur_prob=0.7, 
        motion_blur_prob=0.5, 
        exposure_prob=0.2, 
        gaussian_noise_prob=0.7, 
        flicker_noise_prob=0.3, 
                           ):
    '''
    @param distortions: the distortions to be applied. 
    @param n: the number of parameter groups. 

    Generate n (normalized) parameter groups.  The severity of each 
    distortion should be non-decreasing
    '''
    distortion_strengths = [ 
        list(np.random.uniform(0, 1, size=n)) for _ in distortions ]
    if ordered: 
        for ss in distortion_strengths: 
            ss.sort()
    # Generate random parameter groups
    parameter_groups = []
    for i in range(n):
        parameter_group = {}
        for j, distortion in enumerate(distortions):
            if distortion == 'focus':
                parameter_group['focus'] = {
                    'method': 'gaussian',
                    'radius': distortion_strengths[j][i] if random.random() < focus_blur_prob else 0
                }
            elif distortion == 'motion':
                parameter_group['motion'] = {
                    'method': 'simple',
                    'angle': np.random.uniform(0, 360),
                    'kernel_size': distortion_strengths[j][i] if random.random() < motion_blur_prob else 1,
                    'direction': np.random.uniform(-1, 1)
                }
            elif distortion == 'exposure':
                if random.random() < exposure_prob: 
                    parameter_group['exposure'] = {
                        'alpha': distortion_strengths[j][i],
                        'alpha_type': np.random.choice(['over', 'under']),
                        'beta': np.random.uniform(0, 1),
                        'beta_type': np.random.choice(['over', 'under'])
                    }
            elif distortion == 'noise':
                if random.random() < gaussian_noise_prob: 
                    parameter_group['noise'] = {
                        'flicker_strength': 0,
                        'gaussian_strength': distortion_strengths[j][i]
                    }
            elif distortion == 'jpeg':
                parameter_group['jpeg'] = {
                    'quality': distortion_strengths[j][i]
                }
        # Fill in parameter_group with default value for distortions not presented in the given function argument
        for distortion in ['focus', 'motion', 'exposure', 'noise', 'jpeg']:
            if distortion not in parameter_group:
                if distortion == 'focus':
                    parameter_group['focus'] = {'radius': 0}
                elif distortion == 'motion':
                    parameter_group['motion'] = {'kernel_size': 0}
                elif distortion == 'exposure':
                    parameter_group['exposure'] = None
                elif distortion == 'noise':
                    parameter_group['noise'] = {'flicker_strength': 0, 'gaussian_strength': 0}
                elif distortion == 'jpeg':
                    parameter_group['jpeg'] = {'quality': 0}
        parameter_groups.append(parameter_group)

    return parameter_groups
    

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
