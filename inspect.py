import numpy as np

def inspect_mixedwm38():
    # Load the dataset
    data = np.load('data/raw/MixedWM38.npz')
    
    print("Keys in the .npz file:")
    for key in data.keys():
        print(f"- {key}: shape = {data[key].shape}, dtype = {data[key].dtype}")
        
    # Print sample data for each key
    for key in data.keys():
        print(f"\nSample from {key}:")
        print(data[key][:5] if len(data[key].shape) == 1 else data[key].shape)

if __name__ == "__main__":
    inspect_mixedwm38()