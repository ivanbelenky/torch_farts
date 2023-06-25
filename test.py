import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.functional import F
from torch import nn


from rlud import (
    RLUDConvolver, 
    SeparationCosineLoss, 
    load_img, 
    rect_mask,
    device
)


if __name__ == "__main__":
    img = load_img("../random/imgs/flippy.jpg")
    channels = img.shape[0]
    lebox = [(90,90), (104,104)]
    dataset = RLUDConvolver.weighted_dataset(img, lebox)

    model = RLUDConvolver(channels)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()

    right_tgt, left_tgt, up_tgt, down_tgt = RLUDConvolver.rlud_values(img, lebox, reshape=True)
    right_tgt, left_tgt, up_tgt, down_tgt = right_tgt.to(device), left_tgt.to(device), up_tgt.to(device), down_tgt.to(device)

    y_true, right_x, left_x, up_x, down_x = dataset
    y_true, right_x, left_x, up_x, down_x = y_true.to(device), right_x.to(device), left_x.to(device), up_x.to(device), down_x.to(device)

    rlud_tgt = RLUDConvolver.caterino(right_tgt, left_tgt, up_tgt, down_tgt)
    rlud_x = RLUDConvolver.caterino(right_x, left_x, up_x, down_x)
    
    selected_idxs = RLUDConvolver.cossim_slice_idxs(rlud_x, rlud_tgt, 1024)
    y_true, right_x, left_x, up_x, down_x = y_true[selected_idxs], right_x[selected_idxs], left_x[selected_idxs], up_x[selected_idxs], down_x[selected_idxs]

    dataset_size = len(y_true)
    batch_size = 128
    for epoch in range(10000):
        idxs = np.random.choice(dataset_size, batch_size)
        y_batch, right_batch, left_batch, up_batch, down_batch = y_true[idxs], right_x[idxs], left_x[idxs], up_x[idxs], down_x[idxs]
        
        optimizer.zero_grad()
        y_pred = model(right_batch, left_batch, up_batch, down_batch)

        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch: {} Loss: {}".format(epoch, loss.item()))

    y_pred = model(right_tgt, left_tgt, up_tgt, down_tgt)
    
    original_img = img.permute(1,2,0).numpy()
    mask = rect_mask(original_img.shape, lebox)
    masked_img = original_img*mask
    img_pred = img.clone()
    (lowy, lowx), (upy, upx) = lebox
    img_pred[:, lowy:upy, lowx:upx] = y_pred.detach()
    img_pred = img_pred.permute(1,2,0).numpy()

    fig, ax = plt.subplots(1, 3)
    for a,img,title in zip(ax, [original_img, masked_img, img_pred], ["Original", "Masked", "Reconstructed"]):
        a.imshow(img.astype(np.uint8))
        a.set_xticks([]);a.set_yticks([])
        a.set_title(title)
    plt.show()