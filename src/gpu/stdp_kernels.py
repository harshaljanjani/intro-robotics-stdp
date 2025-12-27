from numba import cuda
import math

@cuda.jit
def _update_traces_kernel(
    trace_pre,
    trace_post,
    spiked_this_step,
    tau_trace_pre,
    tau_trace_post,
    dt
):
    i = cuda.grid(1)
    if i < trace_pre.shape[0]:
        trace_pre[i] *= math.exp(-dt / tau_trace_pre[i])
        trace_post[i] *= math.exp(-dt / tau_trace_post[i])
        if spiked_this_step[i] == 1:
            trace_pre[i] += 1.0
            trace_post[i] += 1.0

def update_traces(
    trace_pre,
    trace_post,
    spiked_this_step,
    tau_trace_pre,
    tau_trace_post,
    dt
):
    threads_per_block = 256
    blocks_per_grid = (trace_pre.shape[0] + (threads_per_block - 1)) // threads_per_block
    _update_traces_kernel[blocks_per_grid, threads_per_block](
        trace_pre,
        trace_post,
        spiked_this_step,
        tau_trace_pre,
        tau_trace_post,
        dt
    )

@cuda.jit
def _update_weights_kernel(
    weights,
    source_neurons,
    target_neurons,
    spiked_this_step,
    trace_pre,
    trace_post,
    learning_rate,
    max_weight
):
    i = cuda.grid(1)
    if i < source_neurons.shape[0]:
        if learning_rate[i] > 0:
            source_id = source_neurons[i]
            target_id = target_neurons[i]
            delta_w = 0.0
            if spiked_this_step[target_id] == 1:
                delta_w += learning_rate[i] * trace_pre[source_id]
            if spiked_this_step[source_id] == 1:
                delta_w -= learning_rate[i] * trace_post[target_id]
            if delta_w != 0.0:
                new_weight = weights[i] + delta_w
                weights[i] = max(0.0, min(new_weight, max_weight[i]))

def update_weights(
    weights,
    source_neurons,
    target_neurons,
    spiked_this_step,
    trace_pre,
    trace_post,
    learning_rate,
    max_weight
):
    threads_per_block = 256
    blocks_per_grid = (source_neurons.shape[0] + (threads_per_block - 1)) // threads_per_block
    _update_weights_kernel[blocks_per_grid, threads_per_block](
        weights,
        source_neurons,
        target_neurons,
        spiked_this_step,
        trace_pre,
        trace_post,
        learning_rate,
        max_weight
    )
