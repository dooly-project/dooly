#!/usr/bin/env python3
"""
Analyze PyTorch trace to extract module-to-kernel execution time mapping.

This script parses PyTorch profiler traces to map model modules (like RowParallelLinear)
to their associated kernel execution times by analyzing CPU annotations and GPU kernel events.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_module_to_kernel(trace_path:str)->Dict:

    with open(trace_path, 'r') as f:
        trace = json.load(f)

    # Collect events by type
    modules = []
    cpu_operations = []
    kernel_events = []
    external_id_to_kernel = {}

    for event in trace.get('traceEvents', []):
        cat = event.get('cat', '')
        name = event.get('name', '')
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        external_id = event.get('args', {}).get('External id')

        if cat == 'python_function':
            if 'nn.Module' in name:
                modules.append({
                    'name': name,
                    'cat': cat,
                    'ts': ts,
                    'dur': dur,
                    'end_ts': ts + dur,
                    'external_id': external_id,
                    'leaf_module': True,
                    'is_layer': 'DecoderLayer_' in name,
                    'layer_num': -1
                })
                
        elif cat == 'cpu_op':
            cpu_operations.append({
                'name': name,
                'cat': cat,
                'ts': ts,
                'dur': dur,
                'end_ts': ts + dur,
                'external_id': external_id
            })
        elif cat == 'kernel':
            kernel_event = {
                'name': name,
                'cat': cat,
                'ts': ts,
                'dur': dur,
                'end_ts': ts + dur,
                'external_id': external_id
            }
            kernel_events.append(kernel_event)

            if external_id is not None:
                if external_id not in external_id_to_kernel:
                    external_id_to_kernel[external_id] = []
                external_id_to_kernel[external_id].append(kernel_event)
    
    leaf_modules = []
    # First pass: Assign layer numbers to all modules based on their parent DecoderLayer
    for module in modules:
        if 'DecoderLayer_' in module['name']:
            # Extract layer number from DecoderLayer
            layer_num = int(module['name'].split('DecoderLayer_')[-1].split('/')[0])
            module_start = module['ts']
            module_end = module['end_ts']

            # Mark all child modules with this layer number
            for child_module in modules:
                if (child_module['ts'] >= module_start and
                    child_module['end_ts'] <= module_end and
                    child_module['name'] != module['name']):
                    child_module['layer_num'] = layer_num

    # Second pass: Find leaf modules
    for module in modules:
        module_start = module['ts']
        module_end = module['end_ts']

        for child_module in modules:
            if (child_module['ts'] >= module_start and
                child_module['end_ts'] <= module_end and
                child_module['name'] != module['name']):
                module['leaf_module'] = False
                break

        if module['leaf_module']:
            leaf_modules.append(module)
            
    
    module_to_kernel = {}
    # Track communication operation counts per layer
    comm_counts_per_layer = {}

    for leaf_module in leaf_modules:
        for cpu_op in cpu_operations:
            if cpu_op['ts'] >= leaf_module['ts'] and cpu_op['end_ts'] <= leaf_module['end_ts']:
                # This CPU op is a child of the module
                if cpu_op['external_id'] is not None:
                    kernels = external_id_to_kernel.get(cpu_op['external_id'])
                    if kernels:
                        # Check if this is a communication operation
                        cpu_op_lower = cpu_op['name'].lower()
                        is_communication = any(comm_op in cpu_op_lower for comm_op in ['all_reduce', 'allreduce', 'all_gather', 'allgather'])

                        if is_communication:
                            # Determine communication type
                            if 'all_reduce' in cpu_op_lower or 'allreduce' in cpu_op_lower:
                                comm_type = 'AllReduce'
                            elif 'all_gather' in cpu_op_lower or 'allgather' in cpu_op_lower:
                                comm_type = 'AllGather'
                            else:
                                comm_type = 'Unknown'

                            # Track occurrence count for this communication type in this layer
                            layer_comm_key = (leaf_module['layer_num'], comm_type)
                            if layer_comm_key not in comm_counts_per_layer:
                                comm_counts_per_layer[layer_comm_key] = 0

                            # Create unique communication module name with occurrence count
                            occurrence = comm_counts_per_layer[layer_comm_key]
                            comm_module_name = f"COMMUNICATION: {comm_type}_{occurrence} (Layer {leaf_module['layer_num']})"
                            comm_counts_per_layer[layer_comm_key] += 1

                            if comm_module_name not in module_to_kernel:
                                module_to_kernel[comm_module_name] = []

                            # Handle all kernels for communication
                            for kernel in kernels:
                                module_to_kernel[comm_module_name].append({
                                    'layer_num': leaf_module['layer_num'],
                                    'cpu_op': cpu_op['name'],
                                    'kernel': kernel['name'],
                                    'kernel_dur_us': kernel['dur']
                                })
                        else:
                            # Regular computation module
                            if leaf_module['name'] not in module_to_kernel:
                                module_to_kernel[leaf_module['name']] = []

                            # Handle all kernels with the same External_id
                            for kernel in kernels:
                                module_to_kernel[leaf_module['name']].append({
                                    'layer_num': leaf_module['layer_num'],
                                    'cpu_op': cpu_op['name'],
                                    'kernel': kernel['name'],
                                    'kernel_dur_us': kernel['dur']
                                })
                            
    print(f"Module-to-kernel mappings: {len(module_to_kernel)}")

    module_total_latencies = {}
    
    for module, ops in module_to_kernel.items():
        total_latency_us = 0
        kernel_count = 0
        
        for op in ops:
            if op['layer_num'] == 0:
                print(f"  {module}: {op['cpu_op']} -> {op['kernel']} ({op['kernel_dur_us']} us)")
                
            total_latency_us += op['kernel_dur_us']
            kernel_count += 1
        
        module_total_latencies[module] = {
            'layer_num': ops[0]['layer_num'] if ops else -1,
            'total_latency_us': total_latency_us,
            'kernel_count': kernel_count,
        }
    
    for module, data in module_total_latencies.items():
        print(f"{data['layer_num']}: {module}: Total kernel latency = {data['total_latency_us']} us over {data['kernel_count']} kernels")

    return module_total_latencies

def extract_module_id(module_name: str) -> int:
    """Extract module ID from names like 'nn.Module: RMSNorm_132'"""
    try:
        if 'nn.Module:' in module_name:
            module_part = module_name.split('nn.Module:')[1].strip()
            if '_' in module_part:
                _, module_id_str = module_part.rsplit('_', 1)
                return int(module_id_str)
    except:
        pass
    return 0

def extract_module_type(module_name: str) -> str:
    """Extract module type from names like 'nn.Module: RMSNorm_132' or 'COMMUNICATION: AllReduce (Layer 5)'"""
    try:
        if 'COMMUNICATION:' in module_name:
            # Extract communication type like "AllReduce" or "AllGather"
            if 'AllReduce' in module_name:
                return 'AllReduce'
            elif 'AllGather' in module_name:
                return 'AllGather'
            else:
                return 'Communication'
        elif 'nn.Module:' in module_name:
            module_part = module_name.split('nn.Module:')[1].strip()
            if '_' in module_part:
                module_type, _ = module_part.rsplit('_', 1)
                return module_type
    except:
        pass
    return 'Unknown'

def compute_median_latencies_by_position(module_total_latencies: Dict) -> Dict:
      """
      Group modules by their execution order position within each layer,
      then compute median latency across all layers for each position.
      """
      # Group modules by layer first (including layer -1)
      layers = {}
      non_layer_modules = []
      for module_name, data in module_total_latencies.items():
          layer_num = data['layer_num']
          if layer_num == -1:  # Non-layer modules (embeddings, logits, etc.)
              non_layer_modules.append((module_name, data))
          else:
              if layer_num not in layers:
                  layers[layer_num] = []
              layers[layer_num].append((module_name, data))

      # Sort modules within each layer by module_id to get execution order
      for layer_num in layers:
          layers[layer_num].sort(key=lambda x: extract_module_id(x[0]))

      # Group by occurrence order within each module type across all layers
      position_groups = {}
      for layer_num, modules_in_layer in layers.items():
          # Track occurrence count for each module type within this layer
          module_type_counts = {}

          for position, (module_name, data) in enumerate(modules_in_layer):
              module_type = extract_module_type(module_name)

              # Count occurrence of this module type in this layer
              if module_type not in module_type_counts:
                  module_type_counts[module_type] = 0
              occurrence = module_type_counts[module_type]
              module_type_counts[module_type] += 1

              # Create position key based on module type and occurrence order
              position_key = f"{module_type}_occurrence_{occurrence}"

              if position_key not in position_groups:
                  position_groups[position_key] = []

              position_groups[position_key].append(data['total_latency_us'])

      # Compute median for each position
      result = {}
      for position_key, latencies in position_groups.items():
          result[position_key] = {
              'median_latency_us': np.median(latencies),
              'median_latency_ms': np.median(latencies) / 1000.0,
              'mean_latency_us': np.mean(latencies),
              'num_layers': len(latencies),
              'layers_with_this_position': sorted(set(layer_num for layer_num, modules_in_layer in layers.items()
                                                      for module_name, data in modules_in_layer
                                                      if extract_module_type(module_name) in position_key))
          }

      # Add non-layer modules (layer -1) as individual entries
      for module_name, data in non_layer_modules:
          module_type = extract_module_type(module_name)
          position_key = f"{module_type}_non_layer"
          result[position_key] = {
              'median_latency_us': data['total_latency_us'],
              'median_latency_ms': data['total_latency_us'] / 1000.0,
              'mean_latency_us': data['total_latency_us'],
              'num_layers': 1,
              'layers_with_this_position': [-1]
          }

      return result


def main():
    parser = argparse.ArgumentParser(description="Analyze module-to-kernel mapping from PyTorch trace")
    parser.add_argument("trace_file", type=str, help="Path to PyTorch trace JSON file")
    parser.add_argument("--output", "-o", type=str, help="Save analysis to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not Path(args.trace_file).exists():
        print(f"Error: Trace file not found: {args.trace_file}")
        return

    print(f"Analyzing trace: {args.trace_file}")

    # Get module to kernel mappings
    module_total_latencies = analyze_module_to_kernel(args.trace_file)

    # Debug: Check communication operations
    print("\nDebugging AllReduce operations:")
    allreduce_modules = [name for name in module_total_latencies.keys() if 'AllReduce' in name]
    for module in sorted(allreduce_modules)[:10]:  # Show first 10
        data = module_total_latencies[module]
        print(f"  Layer {data['layer_num']}: {module} - {data['total_latency_us']} us ({data['kernel_count']} kernels)")
    print(f"  Total AllReduce modules found: {len(allreduce_modules)}")

    # Also check total RowParallelLinear for comparison
    rowparallel_modules = [name for name in module_total_latencies.keys() if 'RowParallelLinear' in name]
    print(f"  Total RowParallelLinear modules found: {len(rowparallel_modules)}")

    # Show layer 0 and layer 1 to see pattern
    print(f"\nLayer 0 modules:")
    for name, data in module_total_latencies.items():
        if data['layer_num'] == 0:
            module_type = extract_module_type(name)
            print(f"    {module_type}")

    print(f"\nLayer 1 modules:")
    for name, data in module_total_latencies.items():
        if data['layer_num'] == 1:
            module_type = extract_module_type(name)
            print(f"    {module_type}")

    result = compute_median_latencies_by_position(module_total_latencies)

    # Print formatted results
    print("\n" + "="*80)
    print("MEDIAN LATENCY BY EXECUTION POSITION")
    print("="*80)

    # Keep original execution order instead of sorting by time
    print(f"{'Position':<40} {'Median (ms)':<12} {'Mean (ms)':<12} {'Layers':<8} {'Layer Range'}")
    print("-" * 100)

    for position_key, data in result.items():
        median_ms = data['median_latency_ms']
        mean_ms = data['mean_latency_us'] / 1000.0
        num_layers = data['num_layers']
        layers_list = data['layers_with_this_position']

        # Show layer range or specific layers
        if len(layers_list) > 5:
            layer_info = f"{min(layers_list)}-{max(layers_list)}"
        else:
            layer_info = str(layers_list)

        print(f"{position_key:<40} {median_ms:<12.3f} {mean_ms:<12.3f} {num_layers:<8} {layer_info}")

    print("\n" + "="*100)

if __name__ == "__main__":
    main()