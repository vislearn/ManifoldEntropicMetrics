import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_probability(z):
    prefac = 1/np.sqrt(2*np.pi)**z.shape[1]
    return prefac*torch.exp(-0.5*torch.sum(z**2, 1))

def contour_grid(z_min, z_max, N_maj, N_min, print_stats=False): # Create contour lines in latent space with N_DIM dimension
    N_majmin = [[] for _ in range(2)]
    for i in range(2):
        N_majmin[i] = N_maj[i]+(N_maj[i] - 1)*N_min[i]
    N_total = np.prod(np.array(N_majmin))

    if print_stats: print('Total number of grid points:', N_total, '\nN per dimension:', N_majmin)

    """
    for i in range(N_DIM):
        z_grids[i] = np.linspace(z_min[i], z_max[i], N_maj[i]+(N_maj[i] - 1)*N_min[i])

    print(z_grids)

    z_grids = np.array(np.meshgrid(*z_grids))
    z_grids = np.moveaxis(z_grids, 0, -1)
    """
    #assert N_DIM == 2
    z_grid0_np = [_ for _ in range(N_maj[1])]
    z_grid1_np = [_ for _ in range(N_maj[0])]

    #1st dim
    for maj in range(N_maj[1]):
        z1_maj_step = (z_max[1]-z_min[1])/(N_maj[1]-1)
        z_grid0_np[maj] = np.array([np.linspace(z_min[0], z_max[0], N_maj[0]+(N_maj[0] - 1)*N_min[0]), np.repeat(z_min[1] + z1_maj_step*maj, N_maj[0]+(N_maj[0] - 1)*N_min[0])])
    #2nd dim
    for maj in range(N_maj[0]):
        z0_maj_step = (z_max[0]-z_min[0])/(N_maj[0]-1)
        z_grid1_np[maj] = np.array([np.repeat(z_min[0] + z0_maj_step*maj, N_maj[0]+(N_maj[1] - 1)*N_min[1]), np.linspace(z_min[1], z_max[1], N_maj[0]+(N_maj[1] - 1)*N_min[1])])
    z_grid0_np = np.array(z_grid0_np).transpose(0,2,1)
    z_grid1_np = np.array(z_grid1_np).transpose(0,2,1)
    return z_grid0_np, z_grid1_np
    
z_min, z_max, N_maj, N_min = [0]*2, [0]*2, [0]*2, [0]*2
z_min[0], z_max[0], N_maj[0], N_min[0] = -2.5, 2.5, 30, 15
z_min[1], z_max[1], N_maj[1], N_min[1] = -2, 2, 8, 50

def plot_contour_grid(model=None, N_DIM=2, BATCHSIZE_samples = 1000, BATCHSIZE_manifolds=100, show_samples=True, show_contour_grid=True, show_original_manifold=False, show_pred_manifold=False, figsize=8, title=None, data_function=None, alpha=1, x_range=[-3, 3], y_range=[-3, 3], N_maj_ticks = 20, N_min_ticks = 10, samples_size=1, manifold_size=3, linewidth=2, pred_manifold_linewidth=3, grid_scaling=1.0, fontsize=None):
    assert model != None, 'No model provided!'

    with torch.no_grad():
        if show_samples: ##Print full distribution
            z = torch.randn(BATCHSIZE_samples, N_DIM).to(device)
            z[:,0] *= grid_scaling
            x, _ = model(z, rev=True)
            samples1 = x.cpu().detach().numpy()
        if show_contour_grid:
            # show contour grid in x-space as a grid transformed from z- to x-space
            z_min, z_max, N_maj, N_min = [0]*2, [0]*2, [0]*2, [0]*2
            z_min[0], z_max[0], N_maj[0], N_min[0] = -2*grid_scaling, 2*grid_scaling, N_maj_ticks, N_min_ticks
            z_min[1], z_max[1], N_maj[1], N_min[1] = -2*grid_scaling, 2*grid_scaling, N_maj_ticks, N_min_ticks #4, 40
            z_grid0_np_temp, z_grid1_np_temp = contour_grid(z_min, z_max, N_maj, N_min)
            
            z_grid0 = torch.tensor(z_grid0_np_temp, dtype=torch.float).to(device)
            z_grid1 = torch.tensor(z_grid1_np_temp, dtype=torch.float).to(device)
            z_grid0_shape, z_grid1_shape = list(z_grid0.shape), list(z_grid1.shape)
            z_grid0, z_grid1 = z_grid0.reshape(-1,2), z_grid1.reshape(-1,2)
            
            x_grid0, _ = model(z_grid0, rev=True)
            x_grid1, _ = model(z_grid1, rev=True)

            samples20 = x_grid0.reshape(*z_grid0_shape).cpu().detach().numpy()
            samples21 = x_grid1.reshape(*z_grid1_shape).cpu().detach().numpy()

            # show contour grid in z-space as a grid transformed from x- to z-space
            x_min, x_max, N_maj, N_min = [0]*2, [0]*2, [0]*2, [0]*2
            x_min[0], x_max[0], N_maj[0], N_min[0] = x_range[0], x_range[1], N_maj_ticks, N_min_ticks
            x_min[1], x_max[1], N_maj[1], N_min[1] = y_range[0], y_range[1], N_maj_ticks, N_min_ticks #4, 40
            x_grid0_np_temp, x_grid1_np_temp = contour_grid(x_min, x_max, N_maj, N_min)
            
            x_gridb0 = torch.tensor(x_grid0_np_temp, dtype=torch.float).to(device)
            x_gridb1 = torch.tensor(x_grid1_np_temp, dtype=torch.float).to(device)
            z_gridb0_shape, z_gridb1_shape = list(x_gridb0.shape), list(x_gridb1.shape)
            x_gridb0, x_gridb1 = x_gridb0.reshape(-1,2), x_gridb1.reshape(-1,2)

        if show_pred_manifold:
            a_min, a_max = -3.0*grid_scaling, 3.0*grid_scaling
            if N_DIM == 2:
                a = torch.linspace(a_min, a_max, BATCHSIZE_manifolds).to(device)
                z_manifold = torch.zeros(BATCHSIZE_manifolds, N_DIM).to(device)
                z_manifold[:,0] = a
            elif N_DIM == 3:
                a_points = int(np.sqrt(BATCHSIZE_manifolds))
                BATCHSIZE_manifolds_temp = a_points**2
                a1 = torch.linspace(a_min, a_max, a_points).to(device)
                a2 = torch.linspace(a_min, a_max, a_points).to(device)
                a1, a2 = torch.meshgrid(a1, a2)
                #print(a1.shape, a2.shape)
                z_manifold = torch.zeros(BATCHSIZE_manifolds_temp, N_DIM).to(device)
                a1 = a1.reshape(BATCHSIZE_manifolds_temp)
                a2 = a2.reshape(BATCHSIZE_manifolds_temp)
                z_manifold[:,0] = a1
                z_manifold[:,1] = a2

            x, _ = model(z_manifold, rev=True)
            samples3 = x.cpu().detach().numpy()
        if show_original_manifold:
            assert data_function != None, 'No data function provided!'
            samples4 = data_function(BATCHSIZE=BATCHSIZE_manifolds) #make_moons(n_samples=BATCHSIZE_manifolds, noise=0)    

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=((1)*figsize, figsize), squeeze=True)
    #plt.tight_layout()

    if x_range is not None:
        axis.set_xlim(x_range)
    if y_range is not None:
        axis.set_ylim(y_range)
    #set aspect ratio

    axis.set_aspect('equal')
    axis.set_facecolor('white')
    if title != None:
        axis.set_title(title, fontsize=fontsize)
    if show_samples:
        axis.scatter(samples1[:,0], samples1[:,1], marker='.', c='black', s=samples_size)
    if show_contour_grid:
        #1st dim:
        for maj in range(N_maj[1]):
            axis.plot(samples20[maj,:,0], samples20[maj,:,1], c='tab:orange', linewidth=linewidth, alpha=alpha)
        #2nd dim:
        for maj in range(N_maj[0]):
            axis.plot(samples21[maj,:,0], samples21[maj,:,1], c='tab:blue', linewidth=linewidth, alpha=alpha)
    if show_pred_manifold:
        axis.plot(samples3[:,0], samples3[:,1], c='red', linewidth=pred_manifold_linewidth)

    if show_original_manifold:
        axis.scatter(samples4[:,0], samples4[:,1], c='yellow', s=manifold_size)
    plt.show()

def plot_pdf(N_samples_manifold, resolution, model, show_original_manifold, show_pred_manifold, data_function=None, N_dim=2, x_range = [-3, 3], y_range = [-3, 3], c_range=None, figsize=6, plot_image=True, save_image=False, save_folder=None, info='', title='', fontsize=None, device=None, **kwargs):
    assert model != None, 'No model provided!'
    try:
        no_plot = kwargs['no_plot']
        fig, axes = kwargs['fig'], kwargs['axes']
    except:
        no_plot = False
    try:
        use_colorbar = kwargs['use_colorbar']
    except:
        use_colorbar = True

    #sample from grid in x-space
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    x, y = torch.meshgrid(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    grid = torch.stack((x, y), 1)

    grid = grid.to(device)
    z_grid, ljd_grid = model(grid) #, return_jacs=False
    if ljd_grid.shape[0] == 1:
        ljd_grid = torch.ones_like(z_grid[:,0]) * ljd_grid
    ljd_grid = ljd_grid.reshape(resolution, resolution)


    pdf_z_grid = 1/np.sqrt(2*np.pi)**N_dim*torch.exp(-0.5*torch.sum(z_grid**2, 1))
    pdf_z_grid = pdf_z_grid.reshape(resolution, resolution)
    pdf_x_grid = pdf_z_grid*torch.exp(ljd_grid) #
    pdf_x_grid = pdf_x_grid.cpu().detach().numpy()

    try:
        constrained_layout = kwargs['constrained_layout']
    except:
        constrained_layout = False

    if not no_plot:
        fig, axes = plt.subplots(1, 1, figsize=(figsize, figsize), squeeze=False, constrained_layout=constrained_layout)
        axes = axes.flatten()

    #Plot PDF of x in x-space
    if c_range == None:
        c_range = [pdf_x_grid.min(), pdf_x_grid.max()]
    
    if kwargs['N_contours'] == None:
        im = axes[0].imshow(pdf_x_grid.T, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='viridis', vmin=c_range[0], vmax=c_range[1])
    #plot contourf
    else:
        im = axes[0].contourf(x.reshape(resolution, resolution).T, y.reshape(resolution, resolution).T, pdf_x_grid.T, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], levels=kwargs['N_contours'], cmap='viridis', vmin=c_range[0], vmax=c_range[1], antialiased=False)
    
    if use_colorbar:
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    #axes[0].set_xlabel('x')
    #axes[0].set_ylabel('y')
    #axes[0].set_title(f'PDF in x-space, {title}')
    axes[0].set_title(title, fontsize=fontsize)

    if show_original_manifold:
        assert data_function != None, 'No data function provided!'
        samples_man = data_function(N_samples_manifold)
        axes[0].scatter(samples_man[:, 0], samples_man[:, 1], s=1, color='yellow', alpha=0.5, label='Original manifold')
    
    if show_pred_manifold:
        z = torch.randn(N_samples_manifold, 2).to(device)
        z[:, 1] = 0
        samples, _ = model(z, rev=True)
        samples = samples.cpu().detach().numpy()
        axes[0].scatter(samples[:, 0], samples[:, 1], s=1, color='red', alpha=0.5, label='Predicted manifold')
    
    axes[0].set_xlim(x_range)
    axes[0].set_ylim(y_range)

    try:
        dpi = kwargs['dpi']
    except:
        dpi = 200
    try:
        axis_off = kwargs['axis_off']
    except:
        axis_off = False
    try:
        output_filetype = kwargs['output_filetype']
    except:
        output_filetype = 'png'
    
    if axis_off:
        axes[0].axis('off')
    
    if not no_plot:
        if save_image:
            assert save_folder is not None
            plt.savefig(os.path.join(save_folder, 'pdf '+info+'.'+output_filetype), transparent=True, dpi=200, bbox_inches='tight')
        if plot_image:
            plt.show()
        plt.close(fig)

def plot_contour_grid_torus(model, kwargs_data, axis, z_dims, x_dims, BATCHSIZE_samples = 1000, show_samples=True, show_contour_grid=True, title=None, alpha=1, x_range=None, y_range=None, N_maj_ticks = 20, N_min_ticks = 10, samples_size=1, linewidth=2,  grid_scaling=1.0, fontsize=None):
    N_dim = kwargs_data['N_dim']
    device = kwargs_data['device']
    data_std = kwargs_data['data_std']
    data_mean = kwargs_data['data_mean']
    rot = kwargs_data['rot']

    with torch.no_grad():
        if show_samples:
            z = torch.randn(BATCHSIZE_samples, N_dim).to(device)
            x, _ = model(z, rev=True)
            x = x * data_std + data_mean
            x = torch.mm(x, rot.t())

            samples1 = x.cpu().detach().numpy()
            samples1 = samples1[:,x_dims]
        if show_contour_grid:
            # show contour grid in x-space as a grid transformed from z- to x-space
            z_min, z_max, N_maj, N_min = [0]*2, [0]*2, [0]*2, [0]*2
            z_min[0], z_max[0], N_maj[0], N_min[0] = -2*grid_scaling, 2*grid_scaling, N_maj_ticks, N_min_ticks
            z_min[1], z_max[1], N_maj[1], N_min[1] = -2*grid_scaling, 2*grid_scaling, N_maj_ticks, N_min_ticks #4, 40
            z_grid0_np_temp, z_grid1_np_temp = contour_grid(z_min, z_max, N_maj, N_min)
            
            z_grid0 = torch.tensor(z_grid0_np_temp, dtype=torch.float).to(device)
            z_grid1 = torch.tensor(z_grid1_np_temp, dtype=torch.float).to(device)

            #pad z_grid with zeros
            z_temp = torch.zeros(z_grid0.shape[0], z_grid0.shape[1], N_dim).to(device)
            z_grid0_temp = z_temp.clone()
            z_grid1_temp = z_temp.clone()
            z_grid0_temp[:,:,z_dims] = z_grid0
            z_grid1_temp[:,:,z_dims] = z_grid1
            z_grid0, z_grid1 = z_grid0_temp, z_grid1_temp
            
            z_grid0_shape, z_grid1_shape = list(z_grid0.shape), list(z_grid1.shape)
            z_grid0, z_grid1 = z_grid0.reshape(-1, N_dim), z_grid1.reshape(-1, N_dim)
            
            x_grid0, _ = model(z_grid0, rev=True)
            x_grid1, _ = model(z_grid1, rev=True)
            x_grid0 = x_grid0 * data_std + data_mean
            x_grid1 = x_grid1 * data_std + data_mean
            x_grid0 = torch.mm(x_grid0, rot.t())
            x_grid1 = torch.mm(x_grid1, rot.t())

            samples20 = x_grid0.reshape(*z_grid0_shape).cpu().detach().numpy()
            samples21 = x_grid1.reshape(*z_grid1_shape).cpu().detach().numpy()
            samples20 = samples20[:,:,x_dims]
            samples21 = samples21[:,:,x_dims]

    if x_range is not None:
        axis.set_xlim(x_range)
    if y_range is not None:
        axis.set_ylim(y_range)
    #set aspect ratio
    axis.set_aspect('equal')
    axis.set_facecolor('white')
    if title != None:
        axis.set_title(title, fontsize=fontsize)
    if show_samples:
        axis.scatter(samples1[:,0], samples1[:,1], marker='.', c='black', s=samples_size, edgecolors='none')
    if show_contour_grid:
        for maj in range(N_maj[1]):
            axis.plot(samples20[maj,:,0], samples20[maj,:,1], c='tab:orange', linewidth=linewidth, alpha=alpha)
        for maj in range(N_maj[0]):
            axis.plot(samples21[maj,:,0], samples21[maj,:,1], c='tab:blue', linewidth=linewidth, alpha=alpha)