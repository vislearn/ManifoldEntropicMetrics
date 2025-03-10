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

def plot_contour_grid(model=None, N_DIM=2, BATCHSIZE_samples = 1000, BATCHSIZE_manifolds=100, show_samples=True, color_samples=False, show_contour_grid=True, show_infinite_grid=False, show_original_manifold=False, show_pred_manifold=False, show_z_space=True, figsize=8, title=None, manifold=None, data_function=None, alpha=1, theta=None, x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3], N_maj_ticks = 20, N_min_ticks = 10, samples_size=1, manifold_size=3, linewidth=2, pred_manifold_linewidth=3, plot_two_sides=False, elev=None, azim=None, x_ortho=False, z_ortho=True, temperature_core=1.0, plot_image=True, save_image=False, save_folder=None, info='', fontsize=None, args=None):
    assert model != None, 'No model provided!'
    if args != None and args['plot'] == True:
        if args['use_grid']:
            #grid samples
            if 'N_grid_samples' in args:
                N_grid_samples = args['N_grid_samples']
            else: N_grid_samples = 200

            x_grid, y_grid = np.meshgrid(np.linspace(*x_range, N_grid_samples), np.linspace(*y_range, N_grid_samples))
            x_grid, y_grid = x_grid.flatten(), y_grid.flatten()
            grid = np.vstack((x_grid,y_grid)).T
            grid = torch.tensor(grid, dtype=torch.float, requires_grad=True).to(device)

            z_grid, ljd_grid = model(grid, rev=False)
            z_grid = z_grid.detach()
            z_grid.requires_grad = True
            x_back, _ = model(z_grid, rev=True)

            grid = grid.cpu().detach().numpy()
            grid = grid.reshape(N_grid_samples, N_grid_samples, 2)
            ljd_grid = ljd_grid.cpu().detach().numpy()
            if args['use_grid_deriv']: ljd_grid_deriv = ljd_grid_deriv.cpu().detach().numpy()

    with torch.no_grad():
        if show_samples: ##Print full distribution
            z = torch.randn(BATCHSIZE_samples, N_DIM).to(device)
            z[:,0] *= temperature_core
            # if N_core_dim == 2:
            #     z[:,1] *= temperature_core
            x, _ = model(z, rev=True) #, return_jacs=False, return_xs=False)
            if color_samples:
                z = z.cpu().detach().numpy()
                r = z[:,0]**2 + z[:,1]**2
                r = r/4
                r = np.where(r>1.0, 1.0, r)
                phi = np.arctan2(z[:,0], z[:,1])/(2*np.pi)+0.5
                a = np.ones((BATCHSIZE_samples, 1, 3))
                a[:,0,0] = phi #H
                a[:,0,1] = r   #S
                c = matplotlib.colors.hsv_to_rgb(a)[:,0,:]
            samples1 = x.cpu().detach().numpy()
        if show_contour_grid:
            # show contour grid in x-space as a grid transformed from z- to x-space
            z_min, z_max, N_maj, N_min = [0]*2, [0]*2, [0]*2, [0]*2
            z_min[0], z_max[0], N_maj[0], N_min[0] = -2*temperature_core, 2*temperature_core, N_maj_ticks, N_min_ticks
            z_min[1], z_max[1], N_maj[1], N_min[1] = -2*temperature_core, 2*temperature_core, N_maj_ticks, N_min_ticks #4, 40
            z_grid0_np_temp, z_grid1_np_temp = contour_grid(z_min, z_max, N_maj, N_min)
            
            z_grid0 = torch.tensor(z_grid0_np_temp, dtype=torch.float).to(device)
            z_grid1 = torch.tensor(z_grid1_np_temp, dtype=torch.float).to(device)
            z_grid0_shape, z_grid1_shape = list(z_grid0.shape), list(z_grid1.shape)
            z_grid0, z_grid1 = z_grid0.reshape(-1,2), z_grid1.reshape(-1,2)
            
            x_grid0, _ = model(z_grid0, rev=True) #, return_jacs=False, return_xs=False)
            x_grid1, _ = model(z_grid1, rev=True) #, return_jacs=False, return_xs=False)

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

            z_gridb0, _ = model(x_gridb0, rev=False) #, return_jacs=False, return_xs=False)
            z_gridb1, _ = model(x_gridb1, rev=False) #, return_jacs=False, return_xs=False)

        if show_infinite_grid:
            z_minb, z_maxb, N_majb, N_minb = [0]*2, [0]*2, [0]*2, [0]*2
            z_minb[0], z_maxb[0], N_majb[0], N_minb[0] = -10, 10, 20, 10
            z_minb[1], z_maxb[1], N_majb[1], N_minb[1] = -10, 10, 20, 10 #4, 40
            z_grid0_np_tempb, z_grid1_np_tempb = contour_grid(z_minb, z_maxb, N_majb, N_minb)
            
            z_grid0b = torch.tensor(z_grid0_np_tempb, dtype=torch.float).to(device)
            z_grid1b = torch.tensor(z_grid1_np_tempb, dtype=torch.float).to(device)
            z_grid0_shapeb, z_grid1_shapeb = list(z_grid0b.shape), list(z_grid1b.shape)
            z_grid0b, z_grid1b = z_grid0b.reshape(-1,2), z_grid1b.reshape(-1,2)

            x_grid0b, _ = model(z_grid0b, rev=True) #, return_jacs=False, return_xs=False)
            x_grid1b, _ = model(z_grid1b, rev=True) #, return_jacs=False, return_xs=False)

            samples20b = x_grid0b.reshape(*z_grid0_shapeb).cpu().detach().numpy()
            samples21b = x_grid1b.reshape(*z_grid1_shapeb).cpu().detach().numpy()
        if show_pred_manifold:
            a_min, a_max = -3.0*temperature_core, 3.0*temperature_core
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

            x, _ = model(z_manifold, rev=True) #, return_jacs=False, return_xs=False)
            samples3 = x.cpu().detach().numpy()
        """
        if show_original_manifold or show_z_space:
            if data_function == None:
                if manifold == None:
                    data_function = lambda BATCHSIZE: make_moons(n_samples=BATCHSIZE, noise=0)[0]
                else:
                    data_function = data_functiontion(manifold, alpha=alpha, theta=theta)
            samples4 = data_function(BATCHSIZE=BATCHSIZE_manifolds) #make_moons(n_samples=BATCHSIZE_manifolds, noise=0)    
           """
        if show_z_space:
            """            
            z_samples1, _ = model(torch.Tensor(samples4).to(device), rev=False) #, return_jacs=False, return_xs=False)
            z_samples1 = z_samples1.cpu().detach().numpy()
            """
            samples5 = torch.Tensor(data_function(BATCHSIZE=BATCHSIZE_samples)).to(device) + torch.randn(BATCHSIZE_samples, N_DIM).to(device)*noise_sigma
            z_samples2, _ = model(samples5, rev=False) #, return_jacs=False, return_xs=False)
            z_samples2 = z_samples2.cpu().detach().numpy()
            if show_contour_grid:
                z_samples30 = z_gridb0.reshape(*z_gridb0_shape).cpu().detach().numpy() #z_grid_0
                z_samples31 = z_gridb1.reshape(*z_gridb1_shape).cpu().detach().numpy() #z_grid_1

    mult_subplots = 0
    if show_z_space:
        mult_subplots += 1
    if N_DIM == 3 and plot_two_sides:
        mult_subplots += 1
    if args != None and args['plot']==True:
        mult_subplots += 1
    
    current_axis = 0
    fig, axes = None, None
    if N_DIM == 2:
        fig, axes = plt.subplots(nrows=1, ncols=1+mult_subplots, figsize=((1+mult_subplots)*figsize, figsize))
        #plt.tight_layout()

        if mult_subplots == 0:
            axis = axes
        else:
            axis = axes[current_axis]
        axis.set_facecolor("black")

    elif N_DIM == 3:
        fig = plt.figure(figsize=((1+mult_subplots)*figsize, figsize))

        axis = fig.add_subplot(1, 1+mult_subplots, 1, projection='3d')
        if x_ortho: axis.set_proj_type('ortho')

        if elev != None or azim != None:
            if elev == None:
                elev = 45
            if azim == None:
                azim = 0
            axis.view_init(elev=elev, azim=azim)

    #axis.set_facecolor("black")

    sides = 1
    if N_DIM == 3 and plot_two_sides:
        sides = 2

    for i in range(sides):
        if N_DIM == 3 and plot_two_sides:
            if i == 0:
                axis.view_init(elev=0, azim=azim)
            elif i == 1:
                axis = fig.add_subplot(1, 1+mult_subplots, 2, projection='3d')
                axis.view_init(elev=90, azim=0)
            axis.set_proj_type('ortho')

        if x_range is not None:
            axis.set_xlim(x_range)
        if y_range is not None:
            axis.set_ylim(y_range)
        if z_range is not None:
            axis.set_zlim(z_range)
        #set aspect ratio
        if N_DIM == 2:
            axis.set_aspect('equal')
            axis.set_facecolor('white')
            if title != None:
                axis.set_title(title, fontsize=fontsize)
        if show_samples:
            if color_samples:
                axis.scatter(samples1[:,0], samples1[:,1], marker='.', c=c, s=samples_size)
            else:
                if N_DIM==2: axis.scatter(samples1[:,0], samples1[:,1], marker='.', c='black', s=samples_size)
                elif N_DIM==3: axis.scatter(samples1[:,0], samples1[:,1], samples1[:,2], marker='.', c='orange', s=samples_size)
        if show_contour_grid or show_infinite_grid:
            #1st dim:
            if show_contour_grid:
                for maj in range(N_maj[1]):
                    axis.plot(samples20[maj,:,0], samples20[maj,:,1], c='tab:orange', linewidth=linewidth, alpha=alpha)
            if show_infinite_grid:
                for maj in range(N_majb[1]):
                    axis.plot(samples20b[maj,:,0], samples20b[maj,:,1], c='tab:orange', linewidth=linewidth/2, alpha=alpha)
            #2nd dim:
            if show_contour_grid:
                for maj in range(N_maj[0]):
                    axis.plot(samples21[maj,:,0], samples21[maj,:,1], c='tab:blue', linewidth=linewidth, alpha=alpha)
            if show_infinite_grid:
                for maj in range(N_majb[1]):
                    axis.plot(samples21b[maj,:,0], samples21b[maj,:,1], c='tab:blue', linewidth=linewidth/2, alpha=alpha)
        if show_pred_manifold:
            if N_DIM==2: axis.plot(samples3[:,0], samples3[:,1], c='red', linewidth=pred_manifold_linewidth)
            elif N_DIM==3:
                #axis.plot3D(samples3[:,0], samples3[:,1], samples3[:,2], c='green', linewidth=linewidth+1)
                #print(samples3.shape)
                X = samples3[:,0].reshape(a_points, a_points)
                Y = samples3[:,1].reshape(a_points, a_points)
                Z = samples3[:,2].reshape(a_points, a_points)
                a_temp = a1.cpu().detach().numpy().reshape(a_points, a_points)
                #print(samples3.shape)

                #norm = plt.Normalize(Z.min(), Z.max())
                #colors = cm.viridis(norm(Z))
                norm = plt.Normalize(a_temp.min(), a_temp.max())
                colors = cm.viridis(norm(a_temp))
                rcount, ccount, _ = colors.shape

                #fig = plt.figure()
                #ax = fig.gca(projection='3d')
                surf = axis.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                                    facecolors=colors, shade=False)
                surf.set_facecolor((0,0,0,0))

                #axis.plot_wireframe(X, Y, Z, linewidth=linewidth) # plot_surface cmap=cm.coolwarm, linewidth=0, antialiased=False
        """
        if show_original_manifold:
            if N_DIM==2: axis.scatter(samples4[:,0], samples4[:,1], c='yellow', s=manifold_size)
            elif N_DIM==3: axis.scatter(samples4[:,0], samples4[:,1], samples4[:,2],  c='yellow', s=manifold_size)
        """
        current_axis += 1

    if args != None and args['plot']==True:
        axis = axes[current_axis]
        if x_range is not None:
            axis.set_xlim(x_range)
        if y_range is not None:
            axis.set_ylim(y_range)
        if z_range is not None:
            axis.set_zlim(z_range)
        #plot grid
        if args['use_grid_deriv']:
            ljd_ampl = np.sqrt(ljd_grid_deriv[:,0]**2 + ljd_grid_deriv[:,1]**2)
            if args['prob_scale_grid_deriv']:
                ljd_ampl = np.log(ljd_ampl+1)
                #ljd_ampl *= get_probability(z_grid).cpu().detach().numpy()
        else:
            ljd_ampl = ljd_grid
            if args['prob_scale_grid_deriv']:
                ljd_ampl *= get_probability(z_grid).cpu().detach().numpy()
        if 'grid_range' in args:
            vmin, vmax = args['grid_range'][0], args['grid_range'][1]
        else:
            vmin, vmax = ljd_ampl.min(), ljd_ampl.max()
        im = axis.imshow(ljd_ampl.reshape(N_grid_samples, N_grid_samples), extent=[*x_range, *y_range], vmin=vmin, vmax=vmax, aspect='auto', origin='lower') #, cmap='RdBu'
        if 'use_colorbar' in args and args['use_colorbar']:
            norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
            #fig.colorbar(cm.ScalarMappable(norm=norm, cmap='RdBu'), cax=axes[current_axis+1], ax=axis)
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.02)
            fig.colorbar(im, cax=cax, orientation='vertical')

        if args['use_arrows']:
            #plot arrows
            ljd_deriv_arrows = ljd_deriv_arrows #/np.sqrt(ljd_deriv_arrows[:,0]**2 + ljd_deriv_arrows[:,1]**2).reshape(-1,1)
            if 'arrow_size' in args:
                arrow_size = args['arrow_size']
            else: arrow_size = 20
            #rescale arrows
            ljd_deriv_arrows_amp = np.sqrt(ljd_deriv_arrows[:,0]**2 + ljd_deriv_arrows[:,1]**2)
            if 'scale_arrows' in args and args['scale_arrows']:
                ljd_deriv_arrows = ljd_deriv_arrows/ljd_deriv_arrows_amp[:,None] * np.log(ljd_deriv_arrows_amp+1).reshape(-1,1)
            axis.quiver(x_arrows[:,0], x_arrows[:,1], ljd_deriv_arrows[:,0], ljd_deriv_arrows[:,1], color='black', scale=arrow_size, width=0.003)
        current_axis += 1

    if show_z_space:
        if N_DIM == 2:
            axis = axes[current_axis]
            axis.set_facecolor("black")
        elif N_DIM == 3:
            axis = fig.add_subplot(1, 1+mult_subplots, 1+mult_subplots, projection='3d')
            if z_ortho: axis.set_proj_type('ortho')
            #ax.set_proj_type('persp', focal_length=1)
            axis.view_init(elev=90, azim=0)
            
            axis.set_zlim([-3, 3])

        axis.set_xlim([-3, 3])
        axis.set_ylim([-3, 3])

        if N_DIM == 2: axis.scatter(z_samples2[:,0], z_samples2[:,1], marker='.', c='gray', s=samples_size) #True manifold
        elif N_DIM == 3: axis.scatter(z_samples2[:,0], z_samples2[:,1], z_samples2[:,2], marker='.', s=samples_size)
        #1st dim:
        if show_contour_grid:
            for maj in range(N_maj[1]):
                axis.plot(z_samples30[maj,:,0], z_samples30[maj,:,1], c='blue', linewidth=linewidth)
        #2nd dim:
        if show_contour_grid:
            for maj in range(N_maj[0]):
                axis.plot(z_samples31[maj,:,0], z_samples31[maj,:,1], c='red', linewidth=linewidth)
        """
        if show_original_manifold: 
            if N_DIM == 2: axis.scatter(z_samples1[:,0], z_samples1[:,1], c='yellow', s=manifold_size) #Inflated manifold = data
            elif N_DIM == 3: axis.scatter(z_samples1[:,0], z_samples1[:,1], z_samples1[:,2], c='yellow', s=manifold_size)
        """
    if save_image:
        assert save_folder is not None
        plt.savefig(os.path.join(save_folder, 'contour_'+info+'.png'), transparent=True, dpi=200, bbox_inches='tight')
    if plot_image: plt.show()
    plt.close(fig)

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