import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def track_causal_graph_snapshot(chg, step):
    snapshot = {
        'step': step,
        'graph': {},
        'num_nodes': 0,
        'num_edges': 0
    }
    if not chg.causal_chains:
        return snapshot
    for cause, effects in chg.causal_chains.items():
        snapshot['graph'][cause] = dict(effects)
        snapshot['num_edges'] += len(effects)
    snapshot['num_nodes'] = len(chg.causal_chains)
    return snapshot

def track_goal_statistics(goal, step):
    if goal is None:
        return None
    return {
        'step': step,
        'type': goal.get('type'),
        'target': goal.get('target'),
        'action': goal.get('action')
    }

def save_causal_learning_plots(causal_history, output_dir, experiment_name="causal_learning"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = plt.figure(figsize=(20, 12))
    # plot 1: motor → object motion strength evolution
    ax1 = plt.subplot(2, 3, 1)
    steps = causal_history['sample_steps']
    for motor_name, strengths in causal_history['motor_to_object_strengths'].items():
        if strengths:
            ax1.plot(steps, strengths, linewidth=2, label=motor_name, marker='o', markersize=4)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Causal Link Strength', fontsize=12)
    ax1.set_title('Learned Motor → Object Motion Causality', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Confidence Threshold')
    # plot 2: causal graph size evolution
    ax2 = plt.subplot(2, 3, 2)
    graph_sizes = causal_history['graph_metrics']
    if graph_sizes:
        nodes = [m['num_nodes'] for m in graph_sizes]
        edges = [m['num_edges'] for m in graph_sizes]
        ax2.plot(steps, nodes, 'b-', linewidth=2, label='Nodes (Populations)', marker='s')
        ax2.plot(steps, edges, 'r-', linewidth=2, label='Edges (Causal Links)', marker='o')
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Causal Graph Growth', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    # plot 3: goal type distribution
    ax3 = plt.subplot(2, 3, 3)
    goal_stats = causal_history['goal_statistics']
    if goal_stats:
        goal_types = [g['type'] for g in goal_stats if g is not None]
        unique_types = list(set(goal_types))
        type_counts = {t: goal_types.count(t) for t in unique_types}
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        ax3.bar(type_counts.keys(), type_counts.values(), color=colors)
        ax3.set_xlabel('Goal Type', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Goal Type Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    # plot 4: uncertainty reduction over time
    ax4 = plt.subplot(2, 3, 4)
    if 'uncertainty_progression' in causal_history and causal_history['uncertainty_progression']:
        uncertainties = causal_history['uncertainty_progression']
        ax4.plot(steps, uncertainties, 'g-', linewidth=2, marker='o')
        ax4.set_xlabel('Step', fontsize=12)
        ax4.set_ylabel('Average Uncertainty', fontsize=12)
        ax4.set_title('Uncertainty Reduction Over Time', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='High Uncertainty Threshold')
        ax4.legend(fontsize=10)
    # plot 5: unique objects encountered
    ax5 = plt.subplot(2, 3, 5)
    if 'unique_objects_over_time' in causal_history and causal_history['unique_objects_over_time']:
        unique_counts = causal_history['unique_objects_over_time']
        ax5.plot(steps, unique_counts, 'purple', linewidth=2, marker='s')
        ax5.set_xlabel('Step', fontsize=12)
        ax5.set_ylabel('Unique Objects Discovered', fontsize=12)
        ax5.set_title('Exploration Progress', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    # plot 6: summary text box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    total_goals = len([g for g in goal_stats if g is not None]) if goal_stats else 0
    approach_goals = len([g for g in goal_stats if g and g['type'] == 'approach']) if goal_stats else 0
    test_goals = len([g for g in goal_stats if g and g['type'] == 'test_causality']) if goal_stats else 0
    explore_goals = len([g for g in goal_stats if g and g['type'] == 'explore_space']) if goal_stats else 0
    final_graph_size = graph_sizes[-1]['num_edges'] if graph_sizes else 0
    final_motor_strengths = {
        name: strengths[-1] if strengths else 0.0
        for name, strengths in causal_history['motor_to_object_strengths'].items()
    }
    summary_text = f"""
    {experiment_name.upper()} SUMMARY
    ═══════════════════════════════════════
    
    Total Steps: {steps[-1] if steps else 0}
    Sampling Interval: {steps[1] - steps[0] if len(steps) > 1 else 'N/A'}
    
    CAUSAL GRAPH:
      Final Nodes: {graph_sizes[-1]['num_nodes'] if graph_sizes else 0}
      Final Edges: {final_graph_size}
      
    FINAL CAUSAL STRENGTHS (Motor→Object):
    """
    for motor_name, strength in final_motor_strengths.items():
        confidence = "HIGH" if strength > 0.7 else "MED" if strength > 0.3 else "LOW"
        summary_text += f"\n      {motor_name}: {strength:.3f} ({confidence})"
    summary_text += f"""
    
    GOAL STATISTICS:
      Total Goals Generated: {total_goals}
      Approach Goals: {approach_goals}
      Test Causality Goals: {test_goals}
      Explore Space Goals: {explore_goals}
    
    LEARNING STATUS:
      {'✓ SUCCESS' if final_graph_size > 0 else '✗ FAILURE'}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.tight_layout()
    plot_path = output_dir / f"{experiment_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[VISUALIZATION] Saved causal learning plot to: {plot_path}")
    # save CSV data.
    csv_path = output_dir / f"{experiment_name}_data_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        header = "Step,NumNodes,NumEdges"
        for motor_name in causal_history['motor_to_object_strengths'].keys():
            header += f",{motor_name}_Strength"
        if 'uncertainty_progression' in causal_history:
            header += ",AvgUncertainty"
        if 'unique_objects_over_time' in causal_history:
            header += ",UniqueObjects"
        f.write(header + "\n")
        for i, step in enumerate(steps):
            row = f"{step}"
            if i < len(graph_sizes):
                row += f",{graph_sizes[i]['num_nodes']},{graph_sizes[i]['num_edges']}"
            else:
                row += ",0,0"
            for motor_name, strengths in causal_history['motor_to_object_strengths'].items():
                if i < len(strengths):
                    row += f",{strengths[i]:.6f}"
                else:
                    row += ",0.0"
            if 'uncertainty_progression' in causal_history and i < len(causal_history['uncertainty_progression']):
                row += f",{causal_history['uncertainty_progression'][i]:.6f}"
            if 'unique_objects_over_time' in causal_history and i < len(causal_history['unique_objects_over_time']):
                row += f",{causal_history['unique_objects_over_time'][i]}"
            f.write(row + "\n")
    print(f"[VISUALIZATION] Saved raw data to: {csv_path}")
    plt.close()

def compute_average_uncertainty(chg, object_tracker, action_space):
    objects = object_tracker.get_all_objects()
    if not objects:
        return 1.0
    uncertainties = []
    for obj_id in objects.keys():
        for action in action_space:
            uncertainty = chg.get_uncertainty(action, obj_id, object_tracker)
            uncertainties.append(uncertainty)
    return np.mean(uncertainties) if uncertainties else 1.0

def print_causal_summary(causal_history):
    print("\n=== CAUSAL LEARNING SUMMARY ===")
    steps = causal_history['sample_steps']
    if not steps:
        print("  [No data collected]")
        return
    print(f"Total sampling points: {len(steps)}")
    print(f"Steps range: {steps[0]} → {steps[-1]}")
    graph_sizes = causal_history['graph_metrics']
    if graph_sizes:
        print(f"\nCausal Graph Growth:")
        print(f"  Initial: {graph_sizes[0]['num_nodes']} nodes, {graph_sizes[0]['num_edges']} edges")
        print(f"  Final: {graph_sizes[-1]['num_nodes']} nodes, {graph_sizes[-1]['num_edges']} edges")
    motor_strengths = causal_history['motor_to_object_strengths']
    print(f"\nMotor → Object Causal Strengths:")
    for motor_name, strengths in motor_strengths.items():
        if strengths:
            initial = strengths[0]
            final = strengths[-1]
            change = final - initial
            print(f"  {motor_name}: {initial:.3f} → {final:.3f} (Δ{change:+.3f})")
    goal_stats = causal_history['goal_statistics']
    if goal_stats:
        valid_goals = [g for g in goal_stats if g is not None]
        print(f"\nGoal Statistics:")
        print(f"  Total goals: {len(valid_goals)}")
        goal_types = [g['type'] for g in valid_goals]
        for gtype in set(goal_types):
            count = goal_types.count(gtype)
            pct = 100.0 * count / len(valid_goals)
            print(f"    {gtype}: {count} ({pct:.1f}%)")
    print("==============================\n")