from tqdm import tqdm

class SliceData:
    def __init__(self, data, slice_len=1024, overlap=0.0, drop_last=True):
        self.data = data
        
        self.slice_len = slice_len
        self.drop_last = drop_last
        self.overlap = overlap
        




    def slice_signal_data(self):
        sliced_data = {}
        for class_name, class_list_data in tqdm(self.data.items(), desc="Slicing signal data"):
            cache_data = []
            for single_signal in tqdm(class_list_data, desc=f"Processing {class_name}", leave=False):
                cache_data.extend(self.slice_single_signal(
                    single_signal, 
                    slice_len=self.slice_len, 
                    overlap=self.overlap, 
                    drop_last=self.drop_last
                    ))
                
            sliced_data[class_name] = cache_data


        self.data = sliced_data
        return sliced_data



    def slice_single_signal(self, signal, slice_len, overlap, drop_last=True):
        """
        对一维信号进行切片（支持重叠）
        :param signal: list 或 1D array
        :param slice_len: 每段长度
        :param overlap: 重叠率 (0~1)
        :param drop_last: 是否丢弃最后不足长度的部分
        """
        assert 0 <= overlap < 1, "overlap 必须在 [0, 1) 之间"

        step = int(slice_len * (1 - overlap))
        if step <= 0:
            raise ValueError("overlap 过大，导致 step <= 0")

        result = []
        n = len(signal)

        for i in range(0, n, step):
            chunk = signal[i:i + slice_len]

            if len(chunk) == slice_len:
                result.append(chunk)
            elif not drop_last:
                result.append(chunk)

        return result




