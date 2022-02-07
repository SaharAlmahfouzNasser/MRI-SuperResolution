from networks import Generator, Discriminator, ProjectionDiscriminator
from networks import init_weights
import torch
from torch import nn
from torch import optim
import dill
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import dataloader
import losses
import numpy as np
import torch.nn.functional as F
import options


# Too many losses to keep track of
# Put everyone in a single place
class BookKeeping:
    def __init__(self, tensorboard_log_path=None, suffix=''):
        self.loss_names = ['mse', 'adversarial',
                           'generator', 'discriminator']
        self.genesis()
        # Initialize tensorboard objects
        self.tboard = dict()
        if tensorboard_log_path is not None:
            if not os.path.exists(tensorboard_log_path):
                os.mkdir(tensorboard_log_path)
            for name in self.loss_names:
                self.tboard[name] = SummaryWriter(os.path.join(tensorboard_log_path, name + '_' + suffix))

    def genesis(self):
        self.losses = {key: 0 for key in self.loss_names}
        self.count = 0

    def update(self, **kwargs):
        for key in kwargs:
            self.losses[key] += kwargs[key]
        self.count += 1

    def reset(self):
        self.genesis()

    def get_avg_losses(self):
        avg_losses = dict()
        for key in self.loss_names:
            avg_losses[key] = self.losses[key] / self.count
        return avg_losses

    def update_tensorboard(self, epoch):
        avg_losses = self.get_avg_losses()
        for key in self.loss_names:
            self.tboard[key].add_scalar(key, avg_losses[key], epoch)


def save_checkpoint(epoch, generator, discriminator, best_metrics, optimizer_G, lr_scheduler_G,
                    optimizer_D, lr_scheduler_D, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch, 'G_state_dict': generator.state_dict(), 'D_state_dict': discriminator.state_dict(),
             'best_metrics': best_metrics, 'optimizer_G': optimizer_G, 'lr_scheduler_G': lr_scheduler_G,
             'optimizer_D': optimizer_D, 'lr_scheduler_D': lr_scheduler_D}
    torch.save(state, filename, pickle_module=dill)


def pbar_desc(label, epoch, total_epochs, loss_val, losses):
    return f'{label}: {epoch:04d}/{total_epochs} | {loss_val:.3f} | mse: {losses["mse"]}'


def save_images(path, lr_images, fake_hr, hr_images, epoch, batchid):

    images_path = os.path.join(path, f'{epoch:04d}')

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for i, tensor in enumerate(lr_images):
        np.save(f'{images_path}/{batchid}_{i:02d}_lr.jpg', tensor.numpy())

    for i, tensor in enumerate(fake_hr):
        np.save(f'{images_path}/{batchid}_{i:02d}_fake.jpg', tensor.numpy())

    for i, tensor in enumerate(hr_images):
        np.save(f'{images_path}/{batchid}_{i:02d}_hr.jpg', tensor.numpy())


def train(G, D, trn_dl, epoch, epochs, MSE, adv_loss, opt_G, opt_D, train_losses, args):
    # Set the nets into training mode
    G.train()
    D.train()

    t_pbar = tqdm(trn_dl, desc=pbar_desc('train', epoch, epochs, 0.0, {'mse': 0.0}))
    for lr_imgs, hr_imgs in t_pbar:

        # Send the images onto the appropriate device
        lr_imgs = lr_imgs.to(args.DEVICE)
        hr_imgs = hr_imgs.to(args.DEVICE)

        # Freeze discriminator, train generator
        for param in D.parameters():
            param.requires_grad = False

        fake_imgs = G(lr_imgs)
        mse_loss = MSE(fake_imgs, hr_imgs)
        mse_display = mse_loss.detach().cpu().item() # F.mse_loss(fake_imgs, hr_imgs).detach().cpu().item()
        # Get predictions from discriminator
        d_fake_preds = D(fake_imgs, lr_imgs)
        # Train the generator to generate fake images
        # such that the discriminator recognizes as real
        g_adv_loss = adv_loss(d_fake_preds, True)

        g_loss = args.MSE_LOSS_WEIGHT * mse_loss + args.ADVERSARIAL_LOSS_WEIGHT * g_adv_loss
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # Unfreeze discriminator, train only the discriminator
        for param in D.parameters():
            param.requires_grad = True

        d_fake_preds = D(fake_imgs.detach(), lr_imgs)  # detach to avoid backprop into G
        d_real_preds = D(hr_imgs, lr_imgs)

        d_loss = adv_loss(d_fake_preds, False) + adv_loss(d_real_preds, True)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        t_pbar.set_description(pbar_desc('train', epoch, args.EPOCHS, g_loss.item(), {'mse': round(mse_display, 3)}))
        train_losses.update(mse=mse_loss.item(), adversarial=g_adv_loss.item(),
                            generator=g_loss.item(), discriminator=d_loss.item())


def evaluate(G, D, val_dl, epoch, epochs, MSE, adv_loss, val_losses, best_val_loss):
    # Set the nets into evaluation mode
    G.eval()
    D.eval()

    v_pbar = tqdm(val_dl, desc=pbar_desc('valid', epoch, epochs, 0.0, {'mse': 0.0}))
    with torch.no_grad():
        for lr_imgs, hr_imgs in v_pbar:
            lr_imgs = lr_imgs.to(args.DEVICE)
            hr_imgs = hr_imgs.to(args.DEVICE)

            fake_imgs = G(lr_imgs)
            mse_loss = MSE(fake_imgs, hr_imgs)
            mse_display = mse_loss.detach().cpu().item() # F.mse_loss(fake_imgs, hr_imgs).detach().cpu().item()
            d_fake_preds = D(fake_imgs, lr_imgs)
            g_adv_loss = adv_loss(d_fake_preds, True)

            g_loss = args.MSE_LOSS_WEIGHT * mse_loss + args.ADVERSARIAL_LOSS_WEIGHT * g_adv_loss

            d_real_preds = D(hr_imgs, lr_imgs)
            d_loss = adv_loss(d_fake_preds, False) + adv_loss(d_real_preds, True)

            val_losses.update(mse=mse_loss.item(), adversarial=g_adv_loss.item(),
                              generator=g_loss.item(), discriminator=d_loss.item())
            v_pbar.set_description(pbar_desc('valid', epoch, args.EPOCHS, g_loss.item(), {'mse': round(mse_display, 3)}))

    # Save best model weights
    avg_val_losses = val_losses.get_avg_losses()
    avg_val_loss = avg_val_losses['generator']
    avg_disval_loss = avg_val_losses['discriminator']
    if avg_val_loss < best_val_loss or epoch % args.SAVE_EVERY == 0:
        best_val_loss = g_loss.item()
        torch.save(G.state_dict(), f'{WEIGHTS_SAVE_PATH}/{args.EXP_NO:02d}-G_epoch-{epoch:04d}_total-loss-{avg_val_loss:.3f}.pth')
        torch.save(D.state_dict(), f'{WEIGHTS_SAVE_PATH}/{args.EXP_NO:02d}-D_epoch-{epoch:04d}_total-loss-{avg_disval_loss:.3f}.pth')

    return best_val_loss


def main(args):
    trn_ds = dataloader.MRIDatasetNpy(args.TRAIN_LR_IMAGES, args.TRAIN_HR_IMAGES, args.LR_NORM_ARR, args.HR_NORM_ARR, norm=False)
    trn_dl = DataLoader(trn_ds, args.TRAIN_BATCH_SIZE, shuffle=True, num_workers=args.WORKERS)

    val_ds = dataloader.MRIDatasetNpy(args.VAL_LR_IMAGES, args.VAL_HR_IMAGES, args.LR_NORM_ARR, args.HR_NORM_ARR, norm=False)
    val_dl = DataLoader(val_ds, args.VAL_BATCH_SIZE, shuffle=False, num_workers=args.WORKERS)
    start_epoch = 1
    best_val_loss = float('inf')

    # Generator
    G = Generator(3, args.IN_CHANNELS, args.OUT_CHANNELS, args.NGF, args.NUM_RESBLOCKS, args.SCHEME, tanh=False)
    print(G)
    print('Generator Parameters:', sum(p.numel() for p in G.parameters()))
    init_weights(G)
    opt_G = optim.Adam(G.parameters(), lr=args.LR_G)
    sched_G = optim.lr_scheduler.StepLR(opt_G, args.LR_STEP, gamma=args.LR_DECAY)

    # Discriminator
    # D = Discriminator(3, IN_CHANNELS, NDF, logits=False)
    D = ProjectionDiscriminator(3, args.IN_CHANNELS, args.NDF, args.SCHEME, logits=False)
    print(D)
    print('Discriminator Parameters:', sum(p.numel() for p in D.parameters()))
    init_weights(D)
    opt_D = optim.Adam(D.parameters(), lr=args.LR_D)
    sched_D = optim.lr_scheduler.StepLR(opt_D, args.LR_STEP, gamma=args.LR_DECAY)

    if not os.path.exists(WEIGHTS_SAVE_PATH):
        os.mkdir(WEIGHTS_SAVE_PATH)

    if args.LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(args.LOAD_CHECKPOINT, pickle_module=dill)
        start_epoch = checkpoint['epoch']

        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])

        opt_G = checkpoint['optimizer_G']
        opt_D = checkpoint['optimizer_D']

        sched_G = checkpoint['lr_scheduler_G']
        sched_D = checkpoint['lr_scheduler_D']

        # best_val_loss = checkpoint['best_metrics']

    G.to(args.DEVICE)
    D.to(args.DEVICE)

    # Losses
    adv_loss = losses.AdversarialLoss(logits=False)
    print(adv_loss)
    adv_loss.to(args.DEVICE)
    MSE = nn.MSELoss()
    # MSE = nn.L1Loss()
    print(MSE)
    MSE.to(args.DEVICE)

    train_losses = BookKeeping(TENSORBOARD_LOGDIR, suffix='trn')
    val_losses = BookKeeping(TENSORBOARD_LOGDIR, suffix='val')

    for epoch in range(start_epoch, args.EPOCHS + 1):

        # Training loop
        train(G, D, trn_dl, epoch, args.EPOCHS, MSE, adv_loss, opt_G, opt_D, train_losses, args)

        # Validation loop
        best_val_loss = evaluate(G, D, val_dl, epoch, args.EPOCHS, MSE, adv_loss, val_losses, best_val_loss, args)

        sched_G.step()
        sched_D.step()

        save_checkpoint(epoch, G, D, None, opt_G, sched_G, opt_D, sched_D)

        train_losses.update_tensorboard(epoch)
        val_losses.update_tensorboard(epoch)

        # Reset all loss for a new epoch
        train_losses.reset()
        val_losses.reset()

        # Save real vs fake samples for quality inspection
        generator = iter(val_dl)
        for j in range(args.BATCHES_TO_SAVE):
            lrs, hrs = next(generator)
            fakes = G(lrs.to(args.DEVICE))

            # Save samples at the end
            save_images(END_EPOCH_SAVE_SAMPLES_PATH, lrs.detach().cpu(), fakes.detach().cpu(), hrs, epoch, j)


if __name__ == '__main__':
    args = options.parse_arguments()
    TENSORBOARD_LOGDIR = f'{args.EXP_NO:02d}-tboard'
    END_EPOCH_SAVE_SAMPLES_PATH = f'{args.EXP_NO:02d}-epoch_end_samples'
    WEIGHTS_SAVE_PATH = f'{args.EXP_NO:02d}-weights'
    main(args)
