import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
from run_nerf_helpers import *
from nerf_renderer import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

tf.compat.v1.enable_eager_execution()

def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, LL=args.LL, use_viewdirs=args.use_viewdirs)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, LL=args.LL, use_viewdirs=args.use_viewdirs)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models

def train():
    parser = config_parser()
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify, rgba=True)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)    

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=8)

        return

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        print('get rays')
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays = np.transpose(rays, [0, 2, 3, 1, 4])
        rays = np.reshape(rays, list(rays.shape[:3]) + [6])
        rays_rgba = np.concatenate([rays, images], -1)
        rays_rgba = np.stack([rays_rgba[i]
                                for i in i_train], axis=0)  # train images only
        rays_rgba = np.reshape(rays_rgba, [-1, 10])
        rays_rgba = rays_rgba.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgba)
        print('done')
        i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.summary.create_file_writer(os.path.join(basedir, 'summaries', expname))

    for i in range(start, N_iters):
        time0 = time.time()
       
        batch = rays_rgba[i_batch:i_batch+N_rand] 
        batch_rays = batch[:, :6]
        target_rgba = batch[:, 6:]

        batch_rays = np.reshape(batch_rays, [-1, 2, 3])
        batch_rays = np.transpose(batch_rays, [1, 0, 2])

        i_batch += N_rand
        if i_batch >= rays_rgba.shape[0]:
            np.random.shuffle(rays_rgba)
            i_batch = 0
        
        #####  Core optimization loop  #####
        
        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGBA.
            rgba = to_rgba(rgb, acc)            
            loss = img2mse(rgba, target_rgba)   
            trans = extras['raw'][..., -1]
            psnr = mse2psnr(loss)            

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                rgba0 = to_rgba(extras['rgb0'], extras['acc0'])
                loss0 = img2mse(rgba0, target_rgba)
                loss += loss0
                psnr0 = mse2psnr(loss0)  

            img_loss = img2mse(rgb, target_rgba[..., :3])
            img_psnr = mse2psnr(img_loss)
            tf.stop_gradient(img_loss)
            tf.stop_gradient(img_psnr)
        
        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))        

        dt = time.time()-time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:

            rgbs, disps = render_path(
                render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                rgbs_still, _ = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0 or i < 10:            
            print(expname, i, psnr.numpy(), loss.numpy(), [img_loss.numpy(), img_psnr.numpy()], global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with writer.as_default():
                tf.summary.scalar('loss', loss, step = i + 1)
                tf.summary.scalar('psnr', psnr, step = i + 1)
                tf.summary.histogram('tran', trans, step = i + 1)
                if args.N_importance > 0:
                    tf.summary.scalar('psnr0', psnr0, step = i + 1)

            if i % args.i_img == 0:

                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3, :4]

                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                **render_kwargs_test)
                rgba = to_rgba(rgb, acc)
                psnr = mse2psnr(img2mse(rgba, target))
                
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i==0:
                    os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgba))
                imageio.imwrite(os.path.join(testimgdir, '{:06d}_acc.png'.format(i)), to8b(acc))

                with writer.as_default():
                    tf.summary.image('rgba', to8b(rgba)[tf.newaxis], step = i + 1)
                    tf.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis], step = i + 1)
                    tf.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis], step = i + 1)

                    tf.summary.scalar('psnr_holdout', psnr, step = i + 1)
                    tf.summary.image('rgb_holdout', target[tf.newaxis], step = i + 1)

                if args.N_importance > 0:
                    with writer.as_default():
                        tf.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis], step = i + 1)
                        tf.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis], step = i + 1)
                        tf.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis], step = i + 1)

        global_step.assign_add(1)


if __name__ == '__main__':
    train()
