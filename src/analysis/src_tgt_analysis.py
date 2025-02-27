import os
import json
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_hidden_states(base_path, tar_dataset_name, src_dataset_name):
    # Load hidden states with and without demonstrations
    with_demo_path = os.path.join(base_path, "hidden_states_with_demo.json")
    without_demo_path = os.path.join(base_path, "hidden_states_without_demo.json")
    
    with open(with_demo_path, 'r') as f:
        hidden_states_with_demo = json.load(f)
    with open(without_demo_path, 'r') as f:
        hidden_states_without_demo = json.load(f)
        
    return hidden_states_with_demo, hidden_states_without_demo

def perform_tsne_visualization(hidden_states_with_demo, hidden_states_without_demo, module, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get number of layers from the data
    num_layers = len(hidden_states_with_demo['0'])
    
    # Process each layer
    for layer in tqdm(range(num_layers), desc="Processing layers"):
        # Collect hidden states for both types
        states_with_demo = []
        states_without_demo = []
        
        # Collect states from all samples
        for idx in hidden_states_with_demo.keys():
            states_with_demo.append(hidden_states_with_demo[idx][str(layer)][module])
            states_without_demo.append(hidden_states_without_demo[idx][str(layer)][module])
            
        # Convert to numpy arrays
        states_with_demo = np.array(states_with_demo)
        states_without_demo = np.array(states_without_demo)
        
        # Combine states and create labels
        combined_states = np.vstack([states_with_demo, states_without_demo])
        labels = np.array(['With Demo'] * len(states_with_demo) + 
                         ['Without Demo'] * len(states_without_demo))
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(combined_states)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                       label=label, alpha=0.6)
            
        plt.title(f't-SNE Visualization - Layer {layer}')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(save_dir, f'layer_{layer}_tsne.png')
        plt.savefig(save_path)
        plt.close()

def main():
    # Configuration
    shot_num = 2  # You can modify this
    shot_method = 'dpp'  # You can modify this
    model_name = 'llama3.1-8b'  # You can modify this
    module = 'hidden'  # You can modify this
    
    # Target and source dataset combinations to analyze
    dataset_pairs = [
        ("arc_challenge", "arc_easy"),
        # ("arc_challenge", "commensense_qa"),
        # ("financial_phrasebank", "sst2")
    ]
    
    for tar_dataset_name, src_dataset_name in dataset_pairs:
        print(f"\nProcessing {tar_dataset_name} -> {src_dataset_name}")
        
        # Construct base path
        base_path = f"data/processed/steering_vector/{shot_num}_shot/{shot_method}/{model_name}/{tar_dataset_name}_{src_dataset_name}"
        
        # Load hidden states
        try:
            hidden_states_with_demo, hidden_states_without_demo = load_hidden_states(
                base_path, tar_dataset_name, src_dataset_name
            )
        except FileNotFoundError:
            print(f"Files not found for {tar_dataset_name}_{src_dataset_name}, skipping...")
            continue
            
        # Create save directory
        save_dir = f"output/analysis/src_tgt_tsne/{shot_num}_shot/{shot_method}/{model_name}/{tar_dataset_name}_{src_dataset_name}"
        
        # Perform t-SNE visualization
        perform_tsne_visualization(
            hidden_states_with_demo,
            hidden_states_without_demo,
            module,
            save_dir
        )
        
        print(f"Visualizations saved to {save_dir}")

if __name__ == "__main__":
    main()
