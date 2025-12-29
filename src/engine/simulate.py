import cupy as cp
import numpy as np
from typing import Dict, List, Tuple
from src.gpu import n_kernels, s_kernels, stdp_kernels

class Simulator:
    def __init__(self, network: Dict[str, cp.ndarray], sim_config: Dict):
        self.network = network
        self.dt = sim_config["dt"]
        self.duration = sim_config["duration"]
        self.num_steps = int(self.duration / self.dt)
        self.current_step = 0
        self.pruning_interval = sim_config.get("pruning_interval", 0)
        self._initialize_spike_buffer()

    def _initialize_spike_buffer(self):
        max_delay = int(cp.max(self.network["delays"]).item()) if self.network["delays"].size > 0 else 0
        self.buffer_size = max_delay + 1
        num_neurons = self.network["membrane_potential"].shape[0]
        self.spike_buffer = cp.zeros((self.buffer_size, num_neurons), dtype=cp.float32)

    def _prune_synapses(self):
        num_synapses_before = self.network['source_neurons'].shape[0]
        if num_synapses_before == 0:
            return
        keep_mask = self.network['weights'] >= self.network['prune_threshold']
        num_synapses_after = int(cp.sum(keep_mask).item())
        if num_synapses_after < num_synapses_before:
            print(f"\n=== Pruning ===")
            print(f"Time: {self.current_step * self.dt:.1f}ms - Synapses before: {num_synapses_before}")
            synaptic_keys = [
                "source_neurons", "target_neurons", "weights", "delays",
                "learning_rate", "max_weight", "prune_threshold"
            ]
            for key in synaptic_keys:
                if key in self.network:
                    self.network[key] = self.network[key][keep_mask]
            print(f"Pruned {num_synapses_before - num_synapses_after} synapses. Synapses after: {num_synapses_after}")
            self._initialize_spike_buffer()
            print(f"=== Pruning Complete ===\n")

    def run(self) -> Tuple[List[float], List[int]]:
        all_spike_times: List[float] = []
        all_spike_indices: List[np.ndarray] = []
        for step in range(self.num_steps):
            self.current_step = step
            # periodically pruned.
            if self.pruning_interval > 0 and step > 0 and step % self.pruning_interval == 0:
                self._prune_synapses()
            buffer_idx = self.current_step % self.buffer_size
            self.network["membrane_potential"] += self.spike_buffer[buffer_idx, :]
            self.spike_buffer[buffer_idx, :] = 0
            spiked_this_step = n_kernels.update_neurons(
                self.network["membrane_potential"],
                self.network["refractory_time"],
                self.network["v_leak"],
                self.network["v_reset"],
                self.network["v_threshold"],
                self.network["tau_m"],
                self.network["i_background"],
                self.dt
            )
            num_spiked = cp.sum(spiked_this_step).item()
            if num_spiked > 0 and self.network['source_neurons'].shape[0] > 0:
                stdp_kernels.update_weights(
                    self.network["weights"],
                    self.network["source_neurons"],
                    self.network["target_neurons"],
                    spiked_this_step,
                    self.network["trace_pre"],
                    self.network["trace_post"],
                    self.network["learning_rate"],
                    self.network["max_weight"]
                )
            stdp_kernels.update_traces(
                self.network["trace_pre"],
                self.network["trace_post"],
                spiked_this_step,
                self.network["tau_trace_pre"],
                self.network["tau_trace_post"],
                self.dt
            )
            if num_spiked > 0 and self.network['source_neurons'].shape[0] > 0:
                spiked_indices = cp.where(spiked_this_step == 1)[0]
                all_spike_times.extend([step * self.dt] * num_spiked)
                all_spike_indices.append(cp.asnumpy(spiked_indices))
                s_kernels.propagate_spikes(
                    spiked_this_step,
                    self.network["source_neurons"],
                    self.network["target_neurons"],
                    self.network["weights"],
                    self.network["delays"],
                    self.spike_buffer,
                    self.current_step
                )
            if step % 1000 == 0:
                current_time_ms = step * self.dt
                print(f"Time: {current_time_ms:.1f}ms - Spiked this step: {num_spiked}")
        # print(f"Final weight: {self.network['weights'][0]}")
        print("\nSimulation finished.")
        return all_spike_times, np.concatenate(all_spike_indices) if all_spike_indices else np.array([])
