import torch 
import time 

def normalize_hip(kps):     
    '''
    Arguments: 
        kps (batch x num_kps x 2): Input 2D normalized joint position (-1 ~ 1)
    Returns: 
        normalized_distance (batch x num_kps) : Normalized Distance from center joint(hip joint) to each joints . 
                                               
    related paper link: https://arxiv.org/pdf/1803.08244.pdf
    '''
    # 1. find center hip coord 
    batch = kps.size(0)
    num_kps = kps.size(1)

    kps_flat = kps.reshape(batch, -1)  # kps_flat : batch x num_kps*2

    # 2. calculate distance of all joints from center hip coord
    l_hip = kps[:, 11, :].clone()      # batch x 2
    r_hip = kps[:, 12, :].clone()      # batch x 2
    cen_hip = ((r_hip + l_hip) / 2).unsqueeze(1).repeat(1, num_kps, 1)      # batch x num_kps x 2
    mean_distance = torch.norm(kps - cen_hip, dim = -1).mean(dim = 1).view(-1, 1)    # batch 224 resolution
    
    # 3. normalize kps. code reference: https://github.com/DwangoMediaVillage/3dpose_gan
    idx = torch.where(mean_distance == 0)
    mean_distance[idx] = 1e-6
    normalized_kps = kps_flat / mean_distance
    
    normalized_hip_x = (normalized_kps[:, 11 * 2] + normalized_kps[:, 12 * 2]) / 2
    normalized_hip_y = (normalized_kps[:, 11 * 2 + 1] + normalized_kps[:, 12 * 2 + 1]) / 2

    normalized_kps[:, 0::2] -= normalized_hip_x.view(-1, 1).repeat(1, num_kps)
    normalized_kps[:, 1::2] -= normalized_hip_y.view(-1, 1).repeat(1, num_kps)

    return normalized_kps   # batch x 17*2


if __name__ == '__main__':
    num_kps = 17
    kps = torch.randn(1, num_kps, 2)
    start = time.time()
    normalized_distance = normalize_hip(kps)
    print(normalized_distance)
    print(f'time {time.time() - start}')


