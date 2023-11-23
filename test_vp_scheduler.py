from diffusers import FlaxScoreSdeVpScheduler, FlaxDDPMScheduler
from jax import random, numpy as jnp
from matplotlib import pyplot as plt

# init ddpm scheduler
ddpm_scheduler = FlaxDDPMScheduler()
ddpm_scheduler.dtype = jnp.float32
ddpm_scheduler_state = ddpm_scheduler.create_state()

# test sampling and adding noise
sample_image = jnp.ones((1,3,32,32))
key_noise = random.PRNGKey(1)
noise = random.normal(key_noise, sample_image.shape, dtype = ddpm_scheduler.dtype)
timesteps = jnp.arange(10, dtype = jnp.int32)*100
T = len(timesteps)*100
# noise added with the linear beta schedule
# add noise function is from the common scheduler class
batch = sample_image.repeat(10,0)
noisy_batch_ddpm = ddpm_scheduler.add_noise(
    ddpm_scheduler_state,
    batch,
    noise,
    timesteps
)
# plot sample diffusion process for ddpm scheduler
fig, axs = plt.subplots(2,11,figsize=(30,5))
noisy_imgs = noisy_batch_ddpm.swapaxes(1,3).swapaxes(1,2)
[axs[0,i].imshow(noisy_imgs[i]) for i in range(10)]
axs[0,-1].plot(noisy_imgs.var(axis=(1,2,3) ), label = "DDPM")
axs[1,-1].plot(noisy_imgs.mean(axis=(1,2,3) ),label = "DDPM")

# init score sde vp scheduler
scheduler = FlaxScoreSdeVpScheduler()
scheduler.dtype = jnp.float32
scheduler_state = scheduler.create_state()

# add noise according to perturbation kernel in Eq. 33 in Song et al
"""
t = jnp.linspace(scheduler.config.sampling_eps, 1, T)
t_batch = t[timesteps]
means, stds = scheduler.marginal_prob(t = t_batch)
noise_batch = noise.repeat(10,0)
noisy_batch_vp = means[:,None,None,None]*batch + stds[:,None,None,None]*noise_batch
noisy_imgs_vp = noisy_batch_vp.swapaxes(1,3).swapaxes(1,2)
"""
noisy_batch_vp = scheduler.add_noise(
    scheduler_state,
    batch,
    noise,
    timesteps
)
noisy_imgs_vp = noisy_batch_vp.swapaxes(1,3).swapaxes(1,2)
# plot sample diffusion process for score sde vp scheduler
[axs[1,i].imshow(noisy_imgs_vp[i]) for i in range(10)]
axs[0,-1].plot(noisy_imgs_vp.var(axis=(1,2,3) ), "--", label = "VP SDE")
axs[1,-1].plot(noisy_imgs_vp.mean(axis=(1,2,3) ), "--", label = "VP SDE")
axs[0,-1].set_title("Variance"); axs[0,-1].legend()
axs[1,-1].set_title("Mean"); axs[1,-1].legend()
fig.show()
