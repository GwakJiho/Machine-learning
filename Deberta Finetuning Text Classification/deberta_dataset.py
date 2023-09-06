import torch
from torch.utils.data import Dataset

class TextClassificationCollator():
    def __init__(self, tokenizer, max_length, with_text=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        if isinstance(samples, dict):
            samples = [samples]
        print(samples)
        texts = [s["input"] for s in samples]
        labels = [s["label"] for s in samples]
        
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding.get('token_type_ids', None),  # 일부 토크나이저는 token_type_ids를 반환하지 않을 수 있습니다.
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        
        if self.with_text:
            return_value['input'] = texts
            
        return return_value



class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        # 슬라이싱을 지원하도록 처리
        if isinstance(index, slice):
            texts_slice = [str(t) for t in self.texts[index]]
            labels_slice = self.labels[index]
            return {
                'input': texts_slice,
                'label': labels_slice
            }

        text = str(self.texts[index])
        label = self.labels[index]
        
        return {
            'input': text,
            'label': label
        }
    

         
