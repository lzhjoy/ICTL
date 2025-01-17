from torch.utils.data import Dataset

class BaseTask(Dataset):
    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name
        self.task_type = None
        self.all_data = None
    
    def download(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        return self.all_data[index]