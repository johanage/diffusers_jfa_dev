# Copyright 2023 Google Brain and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

import math
from jax import random, numpy as jnp
import jax
import flax
import jaxlib

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
     CommonSchedulerState,
     FlaxSchedulerMixin, 
     FlaxSchedulerOutput, 
     broadcast_to_shape_from_left,
     add_noise_common,
     get_velocity_common,
)

from dataclasses import dataclass
from typing import Optional, Tuple, Union

@flax.struct.dataclass
class FlaxScoreSdeVpSchedulerState:
    common              : CommonSchedulerState
    final_alpha_cumprod : jnp.array

    # setable values
    init_noise_sigma    : jaxlib.xla_extension.ArrayImpl
    timesteps           : jaxlib.xla_extension.ArrayImpl
    #t                   : jaxlib.xla_extension.ArrayImpl
    num_inference_steps : Optional[int] = None

    @classmethod
    def create(
        cls,
        common              : CommonSchedulerState,
        final_alpha_cumprod : jaxlib.xla_extension.ArrayImpl,
        init_noise_sigma    : jaxlib.xla_extension.ArrayImpl,
        timesteps           : jaxlib.xla_extension.ArrayImpl,
        #t                   : jaxlib.xla_extension.ArrayImpl,
    ):
        return cls(
            common              = common,
            final_alpha_cumprod = final_alpha_cumprod,
            init_noise_sigma    = init_noise_sigma,
            timesteps           = timesteps,
            #t                   = t,
        )


@dataclass
class FlaxSdeVpOutput(FlaxSchedulerOutput):
    """
    Output class for the ScoreSdeVpScheduler's step function output.

    Args:
        state (`ScoreSdeVpSchedulerState`):
        prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Mean averaged `prev_sample`. Same as `prev_sample`, only mean-averaged over previous timesteps.
    """

    state            : FlaxScoreSdeVpSchedulerState
    prev_sample      : jaxlib.xla_extension.ArrayImpl
    prev_sample_mean : Optional[jaxlib.xla_extension.ArrayImpl] = None

class FlaxScoreSdeVpScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    The variance preserving stochastic differential equation (SDE) scheduler.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more information, see the original paper: https://arxiv.org/abs/2011.13456
    """
    
    #dtype = jnp.dtype
    dtype = jnp.float32

    @property
    def has_state(self):
        return True

    @register_to_config
    def __init__(
        self,
        num_train_timesteps : int                   = 1000,
        snr                 : float                 = 0.15,
        # beta_start and *_end is *_min and *_max divided by num_train_timesteps
        beta_min            : float                 = .1,
        beta_max            : float                 = 20,
        beta_start          : float                 = .1/1000,
        beta_end            : float                 = 20/1000,
        beta_schedule       : str                   = "linear",
        set_alphas_to_one   : bool                  = True,
        # 1e-5 for training and likelihood according to Song et al.
        # 1e-3 for sampling 
        sampling_eps        : float                 = 1e-5, 
        # reduntant?
        trained_betas       : Optional[jaxlib.xla_extension.ArrayImpl] = None,
        steps_offset        : int                   = 0,
    ):
        # reduntant?
        self.sigmas          = None
        self.discrete_sigmas = None
        # TODO figure out if it's possible to just use beta_min and beta_max
        # and then just compute *_start/end
        #self.beta_start = self.beta_min / self.num_train_timesteps
        #self.beta_end   = self.beta_max / self.num_train_timesteps

    def create_state(
        self,
        common : Optional[CommonSchedulerState] = None,
    ) -> FlaxScoreSdeVpSchedulerState:
        
        if common is None:
            common = CommonSchedulerState.create(self)

        final_alpha_cumprod = (
            jnp.array(1.0, dtype=self.dtype) if self.config.set_alphas_to_one else common.alphas_cumprod[0]
        )

        # std of the initial noise distribution
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)
        # timestep indices
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
        #t = jnp.linspace(1, self.sampling_eps, self.num_train_timesteps)
        return FlaxScoreSdeVpSchedulerState.create(
            common              = common,
            final_alpha_cumprod = final_alpha_cumprod,
            init_noise_sigma    = init_noise_sigma,
            timesteps           = timesteps,
            #t                   = t,
        )
    
    def set_timesteps(
        self, 
        state               : FlaxScoreSdeVpSchedulerState,
        num_inference_steps : int, 
        shape               : Tuple = (),
        sampling_eps        : float = None,
    ) -> FlaxScoreSdeVpSchedulerState:

        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        timesteps = jnp.arange(0, num_inference_steps).round()[::-1]
        #t = jnp.linspace(1, sampling_eps, num_inference_steps)
        return state.replace(
            num_inference_steps = num_inference_steps,
            timesteps           = timesteps,
            #t                   = t,
        )

    def _get_variance(
        self, 
        state         : FlaxScoreSdeVpSchedulerState, 
        timestep      : int, 
        prev_timestep : int
    ) -> jnp.float32:
        
        alpha_prod_t = state.common.alphas_cumprod[timestep]
        alpha_prod_t_prev = jnp.where(
            prev_timestep >= 0, state.common.alphas_cumprod[prev_timestep], state.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    """
    -------------------------------------
    Functions used in Predictor START
    -------------------------------------
    """
    def marginal_prob(
        self,
        t : jaxlib.xla_extension.ArrayImpl,
    ) -> Tuple :
        #Log of mean coefficient from Eq. 33 in Song et al.
        a = -0.25 * t**2 * (self.config.beta_max - self.config.beta_min)
        b = -0.5 * t * self.config.beta_min
        log_mean_coeff = a + b
        mean = jnp.exp(log_mean_coeff)
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))
        return (mean, std) 
   
    def sde(
        self,
        t            : jaxlib.xla_extension.ArrayImpl,
        sample       : jaxlib.xla_extension.ArrayImpl,
    ) -> Tuple :
        "Following equation 32 in Song et al."
        beta_t = self.config.beta_start + t * (self.config.beta_end - self.config.beta_start)
        # to get coeffs with same shape as model output (score)
        beta_t = broadcast_to_shape_from_left(beta_t, sample.shape)
        drift = - .5 * beta_t * sample
        diffusion = jnp.sqrt(beta_t)
        return (drift, diffusion)

    

# TODO integrate this into the prediction sampling
    def rsde(
        self,
        t            : jaxlib.xla_extension.ArrayImpl,
        sample       : jaxlib.xla_extension.ArrayImpl,
        model_output : jaxlib.xla_extension.ArrayImpl,
    ) -> Tuple :
        "Create the drift and diffusion functions for the reverse SDE"
        drift, diffusion = self.sde(t, sample)
        drift = drift - diffusion**2 * model_output
        return (drift, diffusion)
    
    def revSDE_discretize(
        self,
        sample       : jaxlib.xla_extension.ArrayImpl,
        t            : jaxlib.xla_extension.ArrayImpl,
        model_output : jaxlib.xla_extension.ArrayImpl,
    ) -> Tuple :
        """
        Doctstring taken from Song's github : https://github.com/yang-song/score_sde/blob/main/sde_lib.py#L129
        Create discretized iteration rules for the reverse diffusion sampler.
        
        Args:
          x: a JAX tensor.
          t: a JAX float representing the time step (from 0 to `self.T`)

        Returns:
          rev_f, rev_G

        """
        f, G = self.discretize(t=t, sample=sample, forward = False, model_output = model_output)
        # Songe et al. eq. 46 App.  App. E
        # Song's code base: rev_f = f - batch_mul(G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.))
        rev_f = f - G**2 * model_output
        # rev_G = jnp.zeros_like(G) if self.probability_flow else G
        rev_G = G
        return rev_f, rev_G

    def discretize(
        self,
        t            : jaxlib.xla_extension.ArrayImpl,
        sample       : jaxlib.xla_extension.ArrayImpl,
        forward      : bool = True,
        model_output : Optional[jaxlib.xla_extension.ArrayImpl] = None,
    ) -> Tuple : 
        """
        Doctstring taken from Song's github : https://github.com/yang-song/score_sde/blob/main/sde_lib.py#L129
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a JAX tensor.
          t: a JAX float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.num_train_timesteps
        # TODO : check whether to include this or not
        #if not forward dt *= -1
        if not forward and model_output is None: 
            raise ValueError("Set model output for reverse mode discretization")
        drift, diffusion = self.sde(t, sample) if forward else self.rsde(t, sample, model_output)
        f = drift * dt
        G = diffusion * jnp.sqrt(dt)
        return f, G
    
    """
    -------------------------------------
    Functions used in Predictor STOP
    -------------------------------------
    """
    
    def step_pred(
        self, 
        state        : FlaxScoreSdeVpSchedulerState,
        model_output : jaxlib.xla_extension.ArrayImpl, 
        timestep     : int,
        sample       : jaxlib.xla_extension.ArrayImpl,
        key          : random.KeyArray, 
        return_dict  : bool = True,
        alg3         : bool = False,
    ) -> Union[FlaxSdeVpOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. 
        Core function to propagate the diffusion process from the learned 
        model outputs (most often the predicted noise).

        Args:
            state (`ScoreSdeVpSchedulerState`): the `FlaxScoreSdeVpScheduler` state data class instance.
            model_output (`jnp.ndarray`): estimated score which is the direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            key: jax.random.KeyArray, seeds to the jax random number generator.
            return_dict (`bool`): option for returning tuple rather than FlaxSdeVpOutput class
            alg3 (`bool`): option for setting the predictor to the predictor used in Alg. 3 of Song et al.

        Returns:
            [`FlaxSdeVpOutput`] or `tuple`: [`FlaxSdeVpOutput`] 
            if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        
        if state.timesteps is None:
            raise ValueError(
                "`state.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )
        
        if alg3:
            """
            ---------------------------------------------------
            Following the specific reverse diffusion sampling found in Alg 3 in App. G in Song et al. 2021
            which is an application of Eq. 46 to Eq. 10 in Song et al. 2021
            Step 3-5 in Alg. 3 PC sampling (VP SDE) in Song et al. 2
            - follows the ancestral sampling scheme i
              line 225 on Song et al.'s github : 
              https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
            ----------------------------------------------------
            """
            # select the beta given timestep index
            beta_t = state.common.betas[timestep]
            # broadcast to match shape of model output
            beta_t = broadcast_to_shape_from_left(beta_t, model_output.shape)
            # last term step 3
            drift = beta_t * model_output
            # step 3 complete equation
            prev_sample_mean = (2 - jnp.sqrt( 1 - beta_t) )*sample + drift
            # step 4 
            #key = random.split(key,num=1)
            noise = random.normal(key=key, shape=sample.shape)
            # coeff last term step 5
            diffusion = jnp.sqrt(beta_t)
            # step 5 
            prev_sample = prev_sample_mean + diffusion * noise
        else:
            """
            Following the general formulation of the reverse diffusion sampling in App. E from Song et al. 2021
            """
            # TODO remove vector init if only one timestep is chosen anyway
            # to save memory
            # define times
            t = jnp.linspace(1, self.sampling_eps, state.num_inference_steps)
            # get mean and std from the marginal probability
            mean, std = self.marginal_prob(t[timestep])
            # normalize the model output to the noise scale'
            # TODO: check if this is correct
            model_output /= std
            # get the drift and diffusion coefficients from the rev SDE (Eq. 46 Songe et al.)
            f, G = self.revSDE_discretize(sample, t[timestep], model_output)
            
            noise = random.normal(key=key, shape=sample.shape)
            prev_sample_mean = sample - f
            prev_sample = prev_sample_mean + G*noise

        # return statement
        if not return_dict:
            return (prev_sample, prev_sample_mean, state)
        return FlaxSdeVpOutput(prev_sample      = prev_sample, 
                               prev_sample_mean = prev_sample_mean, 
                               state            = state)
    
    def step_correct(
        self,
        state        : FlaxScoreSdeVpSchedulerState,
        model_output : jaxlib.xla_extension.ArrayImpl,
        timestep     : int,
        sample       : jaxlib.xla_extension.ArrayImpl,
        key          : random.KeyArray,
        return_dict  : bool = True,
    ) -> Union[FlaxSdeVpOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. 
        This is often run repeatedly after making the prediction for the previous timestep.

        Args:
            state (`ScoreSdeVpSchedulerState`): the `FlaxScoreSdeVpScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            key: jax.random.KeyArray, seeds to the jax random number generator.
            return_dict (`bool`): option for returning tuple rather than FlaxSdeVpOutput class

        Returns:
            [`FlaxSdeVpOutput`] or `tuple`: [`FlaxSdeVpOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if state.num_inference_steps is None:
            raise ValueError(
                "`num_inference_steps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        """
        ---------------------------------------------------------
        Following Alg 3 steps 6-8 described in detail in Alg 5 steps 4-7 in Song et al.
        Notation : from correction Alg 5 from Song et al. : 
        https://arxiv.org/pdf/2011.13456.pdf
        ---------------------------------------------------------
         - noise             -> z (standard Gaussian) (Alg 5 step 4)
         - model_output      -> s_theta^\star(x_i^{j-1}, i) (Alg 5 step 5)
         - grad_norm         -> || g ||_2 (Alg 5 step 6)
         - noise_norm        -> || z ||_2 (Alg 5 step 6) 
         - alpha_i           -> alpha_i (Alg 5 step 6 and alpha_i in first paragraph in subsection 2.2 of Song et al.)
         - step_size        -> epsilon (Alg 5 step 6)
         - prev_sample_mean -> mean(x_i^{j}) = x_i^{j-1} + epsilon*g (Alg 5 step 7)
         - prev_sample      -> x_i^{j} = x_i^{j-1} + epsilon*g +sqrt(2*epsilon)*z (Alg 5 step 7)
        """

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # step 4
        key = random.split(key, num=1)
        noise = random.normal(key=key, shape=sample.shape)

        # step 5 is set model_output to g - this is ignored

        # step 6
        grad_norm = jnp.linalg.norm(model_output)
        noise_norm = jnp.linalg.norm(noise)
        # Song et al. p.16 above Eq. 32 defines beta_i vs. beta(t)
        # note that this is the cumulative product alpha i in Song et al.
        # which is not the same as the alpha_i in the OG Ho et al. 2020 paper
        alpha_i = state.common.alphas_cumprod[timestep]
        # stepsize
        step_size = 2*alpha_i*(self.config.snr * noise_norm / grad_norm) ** 2
        
        # step 7
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample, state)

        return FlaxSdeVpOutput(prev_sample=prev_sample, state=state)
    
    def add_noise(
        self,
        state            : FlaxScoreSdeVpSchedulerState,
        original_samples : jaxlib.xla_extension.ArrayImpl,
        noise            : jaxlib.xla_extension.ArrayImpl,
        timesteps        : jaxlib.xla_extension.ArrayImpl,
    ) -> jaxlib.xla_extension.ArrayImpl:
        return add_noise_common(state.common, original_samples, noise, timesteps)

    def get_velocity(
        self,
        state     : FlaxScoreSdeVpSchedulerState,
        sample    : jaxlib.xla_extension.ArrayImpl,
        noise     : jaxlib.xla_extension.ArrayImpl,
        timesteps : jaxlib.xla_extension.ArrayImpl,
    ) -> jaxlib.xla_extension.ArrayImpl:
        return get_velocity_common(state.common, sample, noise, timesteps)

    def __len__(self):
        return self.config.num_train_timesteps
