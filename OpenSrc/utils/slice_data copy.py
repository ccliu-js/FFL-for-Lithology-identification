from tqdm import tqdm
import random

class SliceData:
    """
    A class to segment long signal data into smaller chunks (slices).
    Supports sliding window slicing with configurable overlap.
    """
    def __init__(self, data, slice_len=1024, overlap=0, drop_last=True, shuffle_random=None):
        """
        Args:
            data (dict): Dictionary where keys are class names and values are lists of signals.
            slice_len (int): The fixed length of each output slice.
            overlap (float): Fraction of overlap between consecutive windows (0.0 to <1.0).
            drop_last (bool): If True, discards segments shorter than slice_len at the end.
            shuffle_random (int): Random seed for shuffling slices within each category.
        """
        
        self.slice_len = slice_len
        self.drop_last = drop_last
        self.overlap = overlap
        self.shuffle_random = shuffle_random

        self.data = self.slice_signal_data(data)



        

    def slice_signal_data(self, data=None):
        """
        Processes nested signal data and organizes slices into a hierarchical dictionary.
        Structure: { speed: { axis: { class: [slices] } } }
        """
        # Use provided input data or fall back to the instance's own data
        data = data if data is not None else self.data
        
        # Initialize the result dictionary
        sliced_data = {}
        
        # Level 1: Iterate through rotation speeds (e.g., 'r4000')
        for speedname, class_speed_data in tqdm(data.items(), desc="Total Progress"):
            if speedname not in sliced_data:
                sliced_data[speedname] = {}
            
            # Level 2: Iterate through sensor axes (e.g., 'Acc_X', 'Acc_Y')
            for axisname, class_axis_data in class_speed_data.items():
                if axisname not in sliced_data[speedname]:
                    sliced_data[speedname][axisname] = {}
                
                # Level 3: Iterate through rock categories (e.g., 'Granite', 'Marble')
                for class_name, class_samples in class_axis_data.items():
                    cache_data = []
                    
                    # Iterate through all raw long-signal samples within this specific category
                    desc_str = f"Slicing {speedname}-{axisname}-{class_name}"
                    for single_signal in tqdm(class_samples, desc=desc_str, leave=False):
                        # Execute slicing logic
                        slices = self.slice_single_signal(
                            single_signal, 
                            slice_len=self.slice_len, 
                            overlap=self.overlap, 
                            drop_last=self.drop_last
                        )
                        cache_data.extend(slices)
                    
                    # Shuffle slices within the same speed, axis, and class to break temporal correlation
                    if self.shuffle_random is not None:
                        random.seed(self.shuffle_random)
                    random.shuffle(cache_data)
                    
                    # Store in the three-level dictionary structure
                    sliced_data[speedname][axisname][class_name] = cache_data
                    
        # Update the instance data and return the structured dictionary
        self.data = sliced_data
        return sliced_data



    def slice_single_signal(self, signal, slice_len, overlap, drop_last=True):
        """
        Performs sliding window slicing on a 1D signal.
        
        Logic:
            step = slice_len * (1 - overlap)
            - If overlap = 0, windows are contiguous (no overlap).
            - If overlap = 0.5, windows overlap by half their length.
        """
        assert 0 <= overlap < 1, "overlap must be in [0, 1)"

        step = int(slice_len * (1 - overlap))
        if step <= 0:
            raise ValueError("overlap is too large, resulting in step <= 0")

        result = []
        n = len(signal)

        for i in range(0, n, step):
            chunk = signal[i:i + slice_len]

            if len(chunk) == slice_len:
                result.append(chunk)
            elif not drop_last:
                result.append(chunk)

        return result




