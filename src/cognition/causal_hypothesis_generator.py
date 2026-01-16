import cupy as cp
from collections import defaultdict
from src.cognition.intuitive_physics_engine import IntuitivePhysicsEngine

class CausalHypothesisGenerator(IntuitivePhysicsEngine):
    def __init__(self, network, pop_info, confidence_threshold=0.3):
        self.network = network
        self.pop_info = pop_info
        self.confidence_threshold = confidence_threshold
        self.causal_chains = defaultdict(dict)
        self.action_to_pop_map = {
            "forward": "Motor_Forward",
            "turn_left": "Motor_Turn_L",
            "turn_right": "Motor_Turn_R"
        }
        print("[COGNITION] CausalHypothesisGenerator initialized (CHG)")

    def _get_connection_strength(self, source_pop, target_pop):
        source = self.pop_info.get(source_pop)
        target = self.pop_info.get(target_pop)
        if source is None or target is None:
            return 0.0
        source_mask = (self.network["source_neurons"] >= source["start"]) & (self.network["source_neurons"] < source["end"])
        target_mask = (self.network["target_neurons"] >= target["start"]) & (self.network["target_neurons"] < target["end"])
        synapse_indices = cp.where(source_mask & target_mask)[0]
        if len(synapse_indices) == 0:
            return 0.0
        avg_weight = cp.mean(self.network["weights"][synapse_indices]).item()
        return max(0.0, avg_weight)

    def extract_direct_links(self):
        direct_links = {}
        all_populations = list(self.pop_info.keys())
        for source in all_populations:
            for target in all_populations:
                if source == target:
                    continue
                strength = self._get_connection_strength(source, target)
                if strength > self.confidence_threshold:
                    direct_links[(source, target)] = strength
        return direct_links

    def build_causal_graph(self):
        direct_links = self.extract_direct_links()
        self.causal_chains = defaultdict(dict)
        for (source, target), strength in direct_links.items():
            self.causal_chains[source][target] = strength
        self._infer_transitive_causality(direct_links)
        return self.causal_chains

    def _infer_transitive_causality(self, direct_links):
        # find chains: if a → b and b → c exist, then a likely causes c
        for (a, b), strength_ab in direct_links.items():
            for (b2, c), strength_bc in direct_links.items():
                if b == b2 and a != c:
                    # transitive link: a → b → c implies a causes c
                    transitive_strength = min(strength_ab, strength_bc) * 0.7
                    if transitive_strength > self.confidence_threshold:
                        existing_strength = self.causal_chains[a].get(c, 0.0)
                        self.causal_chains[a][c] = max(existing_strength, transitive_strength)

    def get_uncertainty(self, action, target_object_id, object_tracker):
        # correctly query the learned graph for uncertainty
        motor_pop = self.action_to_pop_map.get(action)
        if motor_pop is None:
            return 1.0 # high uncertainty for actions we can't model (like 'stop')
        strength = self.causal_chains.get(motor_pop, {}).get("Targeted_Object_Motion", 0.0)
        # normalize strength to be roughly between 0 and 1 for uncertainty calculation
        normalized_strength = min(1.0, strength / 5.0)
        uncertainty = 1.0 - normalized_strength
        return uncertainty

    def print_causal_graph(self):
        print("\n=== CAUSAL GRAPH (CHG) ===")
        if not self.causal_chains:
            print("  [no causal beliefs yet]")
            return
        sorted_chains = sorted(self.causal_chains.items(), key=lambda item: item[0])
        for cause, effects in sorted_chains:
            for effect, strength in sorted(effects.items(), key=lambda x: -x[1]):
                confidence = "HIGH" if strength > 0.7 else "MED" if strength > 0.4 else "LOW"
                print(f"  {cause} → {effect} (strength: {strength:.3f}, conf: {confidence})")
        print("==========================\n")