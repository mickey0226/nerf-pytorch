import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

#x軸を中心に回転
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

#y軸を中心に回転
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    #左手系に変換している
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        #こうやって書くと配列をskipしながら取得できる
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
        #framesに画像に関するさまざまな情報が入っている    
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            #imgsにどんどん画像を追加していく
            imgs.append(imageio.imread(fname))
            #posesに回転行列などを入れていく
            poses.append(np.array(frame['transform_matrix']))
        #画像の次元数としては800*800
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        #countsに訓練データ、検証データ、テストデータの最初のインデックスが入っている
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    #ここで訓練データが入っている要素を示す配列を作っている
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    #すべての画像と姿勢を結合している
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    #camera_angle_xから焦点距離を求めている
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    #４だけ外側に進んで,x軸中心に-30度回転して,ｙ方向にさまざまな角度で回転させる
    #40個のレンダリングの姿勢を作る
    #ここが納得いかないけど先に進んでいく。
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    print("render_poses shape",render_poses.shape)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            #これで画像を半分に縮小することができた
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


